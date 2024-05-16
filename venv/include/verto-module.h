/*
 * Copyright 2011 Red Hat, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*** THE FOLLOWING ARE FOR IMPLEMENTATION MODULES ONLY ***/

#ifndef VERTO_MODULE_H_
#define VERTO_MODULE_H_

#include <verto.h>

#ifndef VERTO_MODULE_TYPES
#define VERTO_MODULE_TYPES
typedef void verto_mod_ctx;
typedef void verto_mod_ev;
#endif

#define VERTO_MODULE_VERSION 3
#define VERTO_MODULE_TABLE(name) verto_module_table_ ## name
#define VERTO_MODULE(name, symb, types) \
    static verto_ctx_funcs name ## _funcs = { \
        name ## _ctx_new, \
        name ## _ctx_default, \
        name ## _ctx_free, \
        name ## _ctx_run, \
        name ## _ctx_run_once, \
        name ## _ctx_break, \
        name ## _ctx_reinitialize, \
        name ## _ctx_set_flags, \
        name ## _ctx_add, \
        name ## _ctx_del \
    }; \
    verto_module VERTO_MODULE_TABLE(name) = { \
        VERTO_MODULE_VERSION, \
        # name, \
        # symb, \
        types, \
        &name ## _funcs, \
    }; \
    verto_ctx * \
    verto_new_ ## name() \
    { \
        return verto_convert(name, 0, NULL); \
    } \
    verto_ctx * \
    verto_default_ ## name() \
    { \
        return verto_convert(name, 1, NULL); \
    }

typedef struct {
    /* Required */ verto_mod_ctx *(*ctx_new)();
    /* Optional */ verto_mod_ctx *(*ctx_default)();
    /* Required */ void (*ctx_free)(verto_mod_ctx *ctx);
    /* Optional */ void (*ctx_run)(verto_mod_ctx *ctx);
    /* Required */ void (*ctx_run_once)(verto_mod_ctx *ctx);
    /* Optional */ void (*ctx_break)(verto_mod_ctx *ctx);
    /* Optional */ void (*ctx_reinitialize)(verto_mod_ctx *ctx);
    /* Optional */ void (*ctx_set_flags)(verto_mod_ctx *ctx,
                                         const verto_ev *ev,
                                         verto_mod_ev *modev);
    /* Required */ verto_mod_ev *(*ctx_add)(verto_mod_ctx *ctx,
                                            const verto_ev *ev,
                                            verto_ev_flag *flags);
    /* Required */ void (*ctx_del)(verto_mod_ctx *ctx,
                                   const verto_ev *ev,
                                   verto_mod_ev *modev);
} verto_ctx_funcs;

typedef struct {
    unsigned int vers;
    const char *name;
    const char *symb;
    verto_ev_type types;
    verto_ctx_funcs *funcs;
} verto_module;

/**
 * Converts an existing implementation specific loop to a verto_ctx.
 *
 * This function also sets the internal default implementation so that future
 * calls to verto_new(NULL) or verto_default(NULL) will use this specific
 * implementation if it was not already set.
 *
 * @param name The name of the module (unquoted)
 * @param deflt Whether the ctx is the default context or not
 * @param ctx The context to store
 * @return A new verto_ctx, or NULL on error. Call verto_free() when done.
 */
#define verto_convert(name, deflt, ctx) \
        verto_convert_module(&VERTO_MODULE_TABLE(name), deflt, ctx)

/**
 * Converts an existing implementation specific loop to a verto_ctx.
 *
 * If you are a module implementation, you probably want the macro above.  This
 * function is generally used directly only when an application is attempting
 * to expose a home-grown event loop to verto.
 *
 * If deflt is non-zero and a default ctx was already defined for this module
 * and ctx is not NULL, than ctx will be free'd and the previously defined
 * default will be returned.
 *
 * If ctx is non-NULL, than the pre-existing verto_mod_ctx will be converted to
 * to a verto_ctx; if deflt is non-zero than this verto_mod_ctx will also be
 * marked as the default loop for this process. If ctx is NULL, than the
 * appropriate constructor will be called: either module->ctx_new() or
 * module->ctx_default() depending on the boolean value of deflt. If
 * module->ctx_default is NULL and deflt is non-zero, than module->ctx_new()
 * will be called and the resulting verto_mod_ctx will be utilized as the
 * default.
 *
 * This function also sets the internal default implementation so that future
 * calls to verto_new(NULL) or verto_default(NULL) will use this specific
 * implementation if it was not already set.
 *
 * @param name The name of the module (unquoted)
 * @param ctx The context private to store
 * @return A new verto_ctx, or NULL on error. Call verto_free() when done.
 */
verto_ctx *
verto_convert_module(const verto_module *module, int deflt, verto_mod_ctx *ctx);

/**
 * Calls the callback of the verto_ev and then frees it via verto_del().
 *
 * The verto_ev is not freed (verto_del() is not called) if it is a signal event.
 *
 * @see verto_add_read()
 * @see verto_add_write()
 * @see verto_add_timeout()
 * @see verto_add_idle()
 * @see verto_add_signal()
 * @see verto_add_child()
 * @see verto_del()
 * @param ev The verto_ev
 */
void
verto_fire(verto_ev *ev);

/**
 * Sets the status of the pid/handle which caused this event to fire.
 *
 * This function does nothing if the verto_ev is not a child type.
 *
 * @see verto_add_child()
 * @param ev The verto_ev to set the status in.
 * @param status The pid/handle status.
 */
void
verto_set_proc_status(verto_ev *ev, verto_proc_status status);

/**
 * Sets the state of the fd which caused this event to fire.
 *
 * This function does nothing if the verto_ev is not a io type.
 *
 * Only the flags VERTO_EV_FLAG_IO_(READ|WRITE|ERROR) are supported. All other
 * flags are unset.
 *
 * @see verto_add_io()
 * @param ev The verto_ev to set the state in.
 * @param state The fd state.
 */
void
verto_set_fd_state(verto_ev *ev, verto_ev_flag state);

#endif /* VERTO_MODULE_H_ */
