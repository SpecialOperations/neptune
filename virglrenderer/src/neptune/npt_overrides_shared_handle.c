/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Shared-HANDLE policy:
 *
 * On macOS, native D3D11 backends (DXMT, D3DMetal) implement linear
 * shm-backed shared 2D textures ('DMTX' v2 POD returned from
 * IDXGIResource::GetSharedHandle and imported via
 * ID3D11Device::OpenSharedResource).  Guest video players such as
 * AVProVideo (MF-MediaEngine-Hardware) decode on a MediaFoundation
 * D3D11 device and share the texture with Unity's D3D11 device inside
 * the SAME process using D3D11_RESOURCE_MISC_SHARED_NTHANDLE +
 * CreateSharedHandle / OpenSharedResource1 (or GetSharedHandle /
 * OpenSharedResource) and optionally IDXGIKeyedMutex.
 *
 * To support same-process cross-device sharing safely without ever
 * exposing raw host pointers to the guest or allowing unchecked
 * handles into the backend:
 *   - CreateTexture2D / CreateTexture2D1 translates SHARED_NTHANDLE
 *     to SHARED (preserving SHARED_KEYEDMUTEX) so native DXMT creates
 *     an shm-backed linear texture instead of failing with E_FAIL.
 *   - GetSharedHandle / CreateSharedHandle exports the host 'DMTX'
 *     descriptor, dups its shm fd into a per-context opaque token
 *     table, and returns an opaque token HANDLE to the guest.
 *   - OpenSharedResource / OpenSharedResource1 validates the token
 *     against the same context's table and imports the shm descriptor
 *     into the target D3D11 device.
 *   - IDXGIKeyedMutex::AcquireSync / ReleaseSync flushes the owning
 *     device's immediate context on ReleaseSync so unified-memory
 *     writes are submitted before the peer device samples them.
 *
 * Unregistered tokens, cross-process handles, named handles, and
 * D3D11 fences still fail with E_INVALIDARG.
 */

#include "npt_overrides.h"
#include "npt_context.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef __APPLE__
#include <fcntl.h>
#endif

#include "c11/threads.h"

#include "neptune-protocol/npt_protocol_defs.h"
#include "neptune-protocol/npt_protocol_host_id3d11device.h"
#include "neptune-protocol/npt_protocol_host_id3d11fence.h"
#include "neptune-protocol/npt_protocol_host_idxgikeyedmutex.h"
#include "neptune-protocol/npt_protocol_host_idxgiresource.h"

#ifdef __APPLE__

#define NPT_DARWIN_SHARED_TEXTURE_MAGIC  0x58544D44u /* 'DMTX' */
#define NPT_DARWIN_SHARED_HANDLE_VERSION 2u
#define NPT_LOCAL_SHARED_MAX_ENTRIES     4096u

struct npt_darwin_shared_texture {
   uint32_t magic;
   uint32_t version;
   int32_t  fd;
   uint32_t width, height;
   uint32_t dxgi_format;
   uint32_t mip_levels, array_size, sample_count;
   uint32_t bind_flags, misc_flags, cpu_access;
   uint64_t stride;
   uint64_t size;
   uint64_t offset;
};

struct npt_local_shared_entry {
   uint64_t token;
   struct npt_darwin_shared_texture desc;
};

struct npt_shared_handle_table {
   mtx_t mutex;
   uint32_t next_seq;
   uint32_t count;
   uint32_t cap;
   struct npt_local_shared_entry *entries;
};

static struct npt_shared_handle_table *
npt_shared_handles_ensure(struct npt_context *ctx)
{
   if (ctx->shared_handles)
      return ctx->shared_handles;
   struct npt_shared_handle_table *t = calloc(1, sizeof(*t));
   if (!t)
      return NULL;
   if (mtx_init(&t->mutex, mtx_plain) != thrd_success) {
      free(t);
      return NULL;
   }
   t->next_seq = 1;
   ctx->shared_handles = t;
   return t;
}

void
npt_shared_handles_fini(struct npt_context *ctx)
{
   struct npt_shared_handle_table *t = ctx->shared_handles;
   if (!t)
      return;
   for (uint32_t i = 0; i < t->count; i++) {
      if (t->entries[i].desc.fd >= 0)
         close(t->entries[i].desc.fd);
   }
   free(t->entries);
   mtx_destroy(&t->mutex);
   free(t);
   ctx->shared_handles = NULL;
}

static uint64_t
npt_shared_handles_insert(struct npt_context *ctx,
                          const struct npt_darwin_shared_texture *src,
                          bool is_nt)
{
   struct npt_shared_handle_table *t = npt_shared_handles_ensure(ctx);
   if (!t)
      return 0;

   int fd_copy = dup(src->fd);
   if (fd_copy < 0)
      return 0;
   int fdflags = fcntl(fd_copy, F_GETFD);
   if (fdflags >= 0)
      fcntl(fd_copy, F_SETFD, fdflags | FD_CLOEXEC);

   mtx_lock(&t->mutex);
   if (t->count >= NPT_LOCAL_SHARED_MAX_ENTRIES) {
      /* Evict oldest slot to prevent unbounded fd accumulation in long sessions. */
      if (t->entries[0].desc.fd >= 0)
         close(t->entries[0].desc.fd);
      memmove(&t->entries[0], &t->entries[1],
              (t->count - 1) * sizeof(t->entries[0]));
      t->count--;
   } else if (t->count == t->cap) {
      uint32_t new_cap = t->cap ? t->cap * 2 : 16;
      struct npt_local_shared_entry *n =
         realloc(t->entries, new_cap * sizeof(*n));
      if (!n) {
         mtx_unlock(&t->mutex);
         close(fd_copy);
         return 0;
      }
      t->entries = n;
      t->cap = new_cap;
   }

   uint32_t seq = t->next_seq++;
   /* Aligned opaque tokens; NT handles stay in low space, KMT handles get high bit. */
   uint64_t token = is_nt ? (0x00100000ull | ((uint64_t)seq << 2))
                          : (0x40000000ull | ((uint64_t)seq << 2));
   struct npt_local_shared_entry *e = &t->entries[t->count++];
   e->token = token;
   e->desc = *src;
   e->desc.fd = fd_copy;
   mtx_unlock(&t->mutex);
   return token;
}

static bool
npt_shared_handles_lookup(struct npt_context *ctx, uint64_t token,
                          struct npt_darwin_shared_texture *out)
{
   struct npt_shared_handle_table *t = ctx->shared_handles;
   if (!t || !token)
      return false;

   bool found = false;
   mtx_lock(&t->mutex);
   for (uint32_t i = 0; i < t->count; i++) {
      if (t->entries[i].token == token) {
         *out = t->entries[i].desc;
         found = true;
         break;
      }
   }
   mtx_unlock(&t->mutex);
   return found;
}

static HRESULT
export_local_shared_handle(struct npt_dispatch_context *dctx,
                           void *dxgi_resource_self,
                           bool is_nt,
                           HANDLE *out_handle)
{
   if (!out_handle)
      return NPT_E_INVALIDARG;
   *out_handle = (HANDLE)0;

   /* Both IDXGIResource and IDXGIResource1 share GetSharedHandle at the same vtable slot. */
   PFN_IDXGIResource_GetSharedHandle get_shared = NPT_COM_VTBL_FUNC(
      PFN_IDXGIResource_GetSharedHandle,
      npt_com_vtable(dxgi_resource_self),
      NPT_VTBL_IDXGIResource_GetSharedHandle);

   HANDLE host_handle = (HANDLE)0;
   HRESULT hr = get_shared(dxgi_resource_self, &host_handle);
   if (NPT_FAILED(hr) || !host_handle) {
      npt_log("shared: local %s host GetSharedHandle failed (hr=0x%x handle=%p)",
              is_nt ? "CreateSharedHandle" : "GetSharedHandle", hr, host_handle);
      return NPT_FAILED(hr) ? hr : NPT_E_INVALIDARG;
   }

   const struct npt_darwin_shared_texture *pod =
      (const struct npt_darwin_shared_texture *)(uintptr_t)host_handle;
   if (pod->magic != NPT_DARWIN_SHARED_TEXTURE_MAGIC ||
       pod->version != NPT_DARWIN_SHARED_HANDLE_VERSION ||
       pod->fd < 0) {
      npt_log("shared: local %s bad DMTX descriptor (magic=0x%x ver=%u fd=%d)",
              is_nt ? "CreateSharedHandle" : "GetSharedHandle",
              pod->magic, pod->version, pod->fd);
      return NPT_E_INVALIDARG;
   }

   uint64_t token = npt_shared_handles_insert(npt_context_from_dispatch(dctx), pod, is_nt);
   if (!token)
      return NPT_E_FAIL;

   *out_handle = (HANDLE)(uintptr_t)token;
   npt_log("shared: local %s %ux%u fmt=%u misc=0x%x -> token=0x%" PRIx64,
           is_nt ? "CreateSharedHandle" : "GetSharedHandle",
           pod->width, pod->height, pod->dxgi_format, pod->misc_flags, token);
   return 0; /* S_OK */
}

static HRESULT
open_local_shared_resource(struct npt_dispatch_context *dctx,
                           void *d3d11_device_self,
                           HANDLE guest_handle,
                           const IID *riid,
                           void **ppResource,
                           const char *method_name)
{
   if (ppResource)
      *ppResource = NULL;
   if (!guest_handle || !riid || !ppResource)
      return NPT_E_INVALIDARG;

   struct npt_darwin_shared_texture desc;
   uint64_t token = (uint64_t)(uintptr_t)guest_handle;
   if (!npt_shared_handles_lookup(npt_context_from_dispatch(dctx), token, &desc)) {
      npt_log("shared: %s unknown token 0x%" PRIx64, method_name, token);
      return NPT_E_INVALIDARG;
   }

   /* Both ID3D11Device and ID3D11Device1 share OpenSharedResource at the same vtable slot. */
   PFN_ID3D11Device_OpenSharedResource open_shared = NPT_COM_VTBL_FUNC(
      PFN_ID3D11Device_OpenSharedResource,
      npt_com_vtable(d3d11_device_self),
      NPT_VTBL_ID3D11Device_OpenSharedResource);

   HRESULT hr = open_shared(d3d11_device_self, (HANDLE)(uintptr_t)&desc, riid, ppResource);
   npt_log("shared: %s token=0x%" PRIx64 " (%ux%u fmt=%u) -> hr=0x%x obj=%p",
           method_name, token, desc.width, desc.height, desc.dxgi_format,
           hr, *ppResource);
   return hr;
}

static void
flush_keyed_mutex_device(void *keyed_mutex_self)
{
   PFN_IDXGIDeviceSubObject_GetDevice get_device = NPT_COM_VTBL_FUNC(
      PFN_IDXGIDeviceSubObject_GetDevice,
      npt_com_vtable(keyed_mutex_self),
      NPT_VTBL_IDXGIDeviceSubObject_GetDevice);

   void *dev = NULL;
   if (NPT_FAILED(get_device(keyed_mutex_self, &NPT_IID_ID3D11Device, &dev)) || !dev)
      return;

   PFN_ID3D11Device_GetImmediateContext get_imm = NPT_COM_VTBL_FUNC(
      PFN_ID3D11Device_GetImmediateContext,
      npt_com_vtable(dev),
      NPT_VTBL_ID3D11Device_GetImmediateContext);

   ID3D11DeviceContext *imm = NULL;
   get_imm(dev, &imm);
   if (imm) {
      PFN_ID3D11DeviceContext_Flush flush = NPT_COM_VTBL_FUNC(
         PFN_ID3D11DeviceContext_Flush,
         npt_com_vtable(imm),
         NPT_VTBL_ID3D11DeviceContext_Flush);
      flush(imm);
      npt_com_release(imm);
   }
   npt_com_release(dev);
}

#else /* !__APPLE__ */

void
npt_shared_handles_fini(UNUSED struct npt_context *ctx)
{
}

#endif /* __APPLE__ */

/* -------------------------------------------------------------------- */
/* Texture2D creation: translate SHARED_NTHANDLE -> SHARED on macOS     */
/* -------------------------------------------------------------------- */

static HRESULT
override_ID3D11Device_CreateTexture2D(
   UNUSED struct npt_dispatch_context *ctx,
   struct npt_command_ID3D11Device_CreateTexture2D *args,
   PFN_ID3D11Device_CreateTexture2D original)
{
#ifdef __APPLE__
   if (args->pDesc && (args->pDesc->MiscFlags & D3D11_RESOURCE_MISC_SHARED_NTHANDLE)) {
      D3D11_TEXTURE2D_DESC desc = *args->pDesc;
      uint32_t orig_misc = desc.MiscFlags;
      desc.MiscFlags &= ~D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
      if (!(desc.MiscFlags & (D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX)))
         desc.MiscFlags |= D3D11_RESOURCE_MISC_SHARED;
      args->ret = original(args->_self, &desc, args->pInitialData, args->ppTexture2D);
      npt_log("shared: CreateTexture2D %ux%u fmt=%u misc=0x%x->0x%x -> hr=0x%x",
              desc.Width, desc.Height, desc.Format, orig_misc, desc.MiscFlags, args->ret);
      return args->ret;
   }
#endif
   args->ret = original(args->_self, args->pDesc, args->pInitialData, args->ppTexture2D);
   return args->ret;
}

static HRESULT
override_ID3D11Device3_CreateTexture2D1(
   UNUSED struct npt_dispatch_context *ctx,
   struct npt_command_ID3D11Device3_CreateTexture2D1 *args,
   PFN_ID3D11Device3_CreateTexture2D1 original)
{
#ifdef __APPLE__
   if (args->pDesc1 && (args->pDesc1->MiscFlags & D3D11_RESOURCE_MISC_SHARED_NTHANDLE)) {
      D3D11_TEXTURE2D_DESC1 desc = *args->pDesc1;
      uint32_t orig_misc = desc.MiscFlags;
      desc.MiscFlags &= ~D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
      if (!(desc.MiscFlags & (D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX)))
         desc.MiscFlags |= D3D11_RESOURCE_MISC_SHARED;
      args->ret = original(args->_self, &desc, args->pInitialData, args->ppTexture2D);
      npt_log("shared: CreateTexture2D1 %ux%u fmt=%u misc=0x%x->0x%x -> hr=0x%x",
              desc.Width, desc.Height, desc.Format, orig_misc, desc.MiscFlags, args->ret);
      return args->ret;
   }
#endif
   args->ret = original(args->_self, args->pDesc1, args->pInitialData, args->ppTexture2D);
   return args->ret;
}

/* -------------------------------------------------------------------- */
/* Outbound: Create*SharedHandle / GetSharedHandle                      */
/* -------------------------------------------------------------------- */

static HRESULT
override_IDXGIResource_GetSharedHandle(
   struct npt_dispatch_context *ctx,
   struct npt_command_IDXGIResource_GetSharedHandle *args,
   UNUSED PFN_IDXGIResource_GetSharedHandle original)
{
#ifdef __APPLE__
   args->ret = export_local_shared_handle(ctx, args->_self, false, args->pSharedHandle);
   return args->ret;
#else
   if (args->pSharedHandle)
      *args->pSharedHandle = (HANDLE)0;
   args->ret = NPT_E_INVALIDARG;
   return args->ret;
#endif
}

static HRESULT
override_IDXGIResource1_CreateSharedHandle(
   struct npt_dispatch_context *ctx,
   struct npt_command_IDXGIResource1_CreateSharedHandle *args,
   UNUSED PFN_IDXGIResource1_CreateSharedHandle original)
{
#ifdef __APPLE__
   if (args->lpName) {
      if (args->pHandle)
         *args->pHandle = (HANDLE)0;
      args->ret = NPT_E_INVALIDARG;
      return args->ret;
   }
   args->ret = export_local_shared_handle(ctx, args->_self, true, args->pHandle);
   return args->ret;
#else
   if (args->pHandle)
      *args->pHandle = (HANDLE)0;
   args->ret = NPT_E_INVALIDARG;
   return args->ret;
#endif
}

static HRESULT
reject_ID3D11Fence_CreateSharedHandle(
   UNUSED struct npt_dispatch_context *ctx,
   struct npt_command_ID3D11Fence_CreateSharedHandle *args,
   UNUSED PFN_ID3D11Fence_CreateSharedHandle original)
{
   if (args->pHandle)
      *args->pHandle = (HANDLE)0;
   args->ret = NPT_E_INVALIDARG;
   return args->ret;
}

/* -------------------------------------------------------------------- */
/* Inbound: OpenShared*                                                 */
/* -------------------------------------------------------------------- */

static HRESULT
override_ID3D11Device_OpenSharedResource(
   struct npt_dispatch_context *ctx,
   struct npt_command_ID3D11Device_OpenSharedResource *args,
   UNUSED PFN_ID3D11Device_OpenSharedResource original)
{
#ifdef __APPLE__
   args->ret = open_local_shared_resource(
      ctx, args->_self, args->hResource, args->ReturnedInterface,
      args->ppResource, "OpenSharedResource");
   return args->ret;
#else
   if (args->ppResource)
      *args->ppResource = NULL;
   args->ret = NPT_E_INVALIDARG;
   return args->ret;
#endif
}

static HRESULT
override_ID3D11Device1_OpenSharedResource1(
   struct npt_dispatch_context *ctx,
   struct npt_command_ID3D11Device1_OpenSharedResource1 *args,
   UNUSED PFN_ID3D11Device1_OpenSharedResource1 original)
{
#ifdef __APPLE__
   args->ret = open_local_shared_resource(
      ctx, args->_self, args->hResource, args->returnedInterface,
      args->ppResource, "OpenSharedResource1");
   return args->ret;
#else
   if (args->ppResource)
      *args->ppResource = NULL;
   args->ret = NPT_E_INVALIDARG;
   return args->ret;
#endif
}

static HRESULT
reject_ID3D11Device1_OpenSharedResourceByName(
   UNUSED struct npt_dispatch_context *ctx,
   struct npt_command_ID3D11Device1_OpenSharedResourceByName *args,
   UNUSED PFN_ID3D11Device1_OpenSharedResourceByName original)
{
   if (args->ppResource)
      *args->ppResource = NULL;
   args->ret = NPT_E_INVALIDARG;
   return args->ret;
}

static HRESULT
reject_ID3D11Device5_OpenSharedFence(
   UNUSED struct npt_dispatch_context *ctx,
   struct npt_command_ID3D11Device5_OpenSharedFence *args,
   UNUSED PFN_ID3D11Device5_OpenSharedFence original)
{
   if (args->ppFence)
      *args->ppFence = NULL;
   args->ret = NPT_E_INVALIDARG;
   return args->ret;
}

/* -------------------------------------------------------------------- */
/* IDXGIKeyedMutex: native DXMT stubs AcquireSync/ReleaseSync           */
/* -------------------------------------------------------------------- */

static HRESULT
override_IDXGIKeyedMutex_AcquireSync(
   UNUSED struct npt_dispatch_context *ctx,
   struct npt_command_IDXGIKeyedMutex_AcquireSync *args,
   PFN_IDXGIKeyedMutex_AcquireSync original)
{
#ifdef __APPLE__
   static bool logged;
   if (!logged) {
      logged = true;
      npt_log("shared: IDXGIKeyedMutex::AcquireSync active (key=%" PRIu64 ")",
              (uint64_t)args->Key);
   }
   (void)original;
   args->ret = 0; /* S_OK */
   return args->ret;
#else
   args->ret = original(args->_self, args->Key, args->dwMilliseconds);
   return args->ret;
#endif
}

static HRESULT
override_IDXGIKeyedMutex_ReleaseSync(
   UNUSED struct npt_dispatch_context *ctx,
   struct npt_command_IDXGIKeyedMutex_ReleaseSync *args,
   PFN_IDXGIKeyedMutex_ReleaseSync original)
{
#ifdef __APPLE__
   (void)original;
   flush_keyed_mutex_device(args->_self);
   args->ret = 0; /* S_OK */
   return args->ret;
#else
   args->ret = original(args->_self, args->Key);
   return args->ret;
#endif
}

/* -------------------------------------------------------------------- */
/* Override tables                                                      */
/* -------------------------------------------------------------------- */

struct npt_dispatch_idxgiresource_overrides npt_idxgiresource_overrides = {
   .GetSharedHandle = override_IDXGIResource_GetSharedHandle,
};

struct npt_dispatch_idxgiresource1_overrides npt_idxgiresource1_overrides = {
   .CreateSharedHandle = override_IDXGIResource1_CreateSharedHandle,
};

struct npt_dispatch_idxgikeyedmutex_overrides npt_idxgikeyedmutex_overrides = {
   .AcquireSync = override_IDXGIKeyedMutex_AcquireSync,
   .ReleaseSync = override_IDXGIKeyedMutex_ReleaseSync,
};

struct npt_dispatch_id3d11device_overrides npt_id3d11device_overrides = {
   .CreateTexture2D = override_ID3D11Device_CreateTexture2D,
   .OpenSharedResource = override_ID3D11Device_OpenSharedResource,
};

struct npt_dispatch_id3d11device1_overrides npt_id3d11device1_overrides = {
   .OpenSharedResource1 = override_ID3D11Device1_OpenSharedResource1,
   .OpenSharedResourceByName = reject_ID3D11Device1_OpenSharedResourceByName,
};

struct npt_dispatch_id3d11device3_overrides npt_id3d11device3_overrides = {
   .CreateTexture2D1 = override_ID3D11Device3_CreateTexture2D1,
};

struct npt_dispatch_id3d11device5_overrides npt_id3d11device5_overrides = {
   .OpenSharedFence = reject_ID3D11Device5_OpenSharedFence,
};

struct npt_dispatch_id3d11fence_overrides npt_id3d11fence_overrides = {
   .CreateSharedHandle = reject_ID3D11Fence_CreateSharedHandle,
};
