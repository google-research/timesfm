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

#ifndef VERTO_H_
#define VERTO_H_

#include <time.h>   /* For time_t */
#include <unistd.h> /* For pid_t */

#ifdef WIN32
#include <windows.h>
typedef HANDLE verto_proc;
typedef DWORD verto_proc_status;
#else
#include <sys/types.h>
typedef pid_t verto_proc;
typedef int verto_proc_status;
#endif

#define VERTO_SIG_IGN ((verto_callback *) 1)

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

typedef struct verto_ctx verto_ctx;
typedef struct verto_ev verto_ev;

typedef enum {
    VERTO_EV_TYPE_NONE = 0,
    VERTO_EV_TYPE_IO = 1,
    VERTO_EV_TYPE_TIMEOUT = 1 << 1,
    VERTO_EV_TYPE_IDLE = 1 << 2,
    VERTO_EV_TYPE_SIGNAL = 1 << 3,
    VERTO_EV_TYPE_CHILD = 1 << 4
} verto_ev_type;

typedef enum {
    VERTO_EV_FLAG_NONE = 0,
    VERTO_EV_FLAG_PERSIST = 1,
    VERTO_EV_FLAG_PRIORITY_LOW = 1 << 1,
    VERTO_EV_FLAG_PRIORITY_MEDIUM = 1 << 2,
    VERTO_EV_FLAG_PRIORITY_HIGH = 1 << 3,
    VERTO_EV_FLAG_IO_READ = 1 << 4,
    VERTO_EV_FLAG_IO_WRITE = 1 << 5,
    VERTO_EV_FLAG_IO_ERROR = 1 << 7,
    VERTO_EV_FLAG_IO_CLOSE_FD = 1 << 8,
    VERTO_EV_FLAG_REINITIABLE = 1 << 6,
    _VERTO_EV_FLAG_MUTABLE_MASK = VERTO_EV_FLAG_PRIORITY_LOW
                                  | VERTO_EV_FLAG_PRIORITY_MEDIUM
                                  | VERTO_EV_FLAG_PRIORITY_HIGH
                                  | VERTO_EV_FLAG_IO_READ
                                  | VERTO_EV_FLAG_IO_WRITE,
    _VERTO_EV_FLAG_MAX = VERTO_EV_FLAG_IO_CLOSE_FD
} verto_ev_flag;

typedef void (verto_callback)(verto_ctx *ctx, verto_ev *ev);

/**
 * Creates a new event context using an optionally specified implementation
 * and/or optionally specified required features.
 *
 * If you are an application that has already decided on using a particular
 * event loop implementation, you should not call this function, but instead
 * import the verto-NAME.h header and link against the verto-NAME.so, where
 * NAME is the implementation you wish to use.
 *
 * If you are a library, you should generally avoid creating event contexts
 * on your own but allow applications to pass in a verto_ctx you can use.
 *
 * There are two cases where you should use this function.  The first is
 * where you have a need to choose an implementation at run time, usually
 * for testing purposes.  The second and more common is when you simply
 * wish to remain implementation agnostic.  In this later case, you should
 * always call like this: verto_new(NULL, ...).  This lets verto choose the best
 * implementation to use.
 *
 * If impl is not NULL, a new context is returned which is backed by the
 * implementation specified. If the implementation specified is not
 * available or if the required types (reqtypes) are not provided by the
 * named implementation, NULL is returned. The parameter 'impl' can specify:
 *   * The full path to an implementation library
 *   * The name of the implementation library (i.e. - "glib" or "libev")
 *
 * If impl is NULL, verto will attempt to automatically determine the
 * best implementation to use.
 *
 * First, verto will attempt to use an existing, previously loaded
 * implementation. This is handled automatically by internal caching of either
 * the first implementation loaded or the one specified by verto_set_default().
 *
 * Second, verto will attempt to discern if you are already linked to any
 * of the supported implementations (to avoid wasting memory by loading
 * extra unnecessary libraries).  If you are linked to one supported
 * implementation, that implementation will be chosen.  If you are linked
 * to more than one supported implementation one of the ones linked to
 * will be chosen, but the order of the particular choice is undefined.
 *
 * Third, verto will attempt to load the compile-time default, if defined at
 * build time and available at runtime.
 *
 * Last, verto will attempt to load any implementation installed. The specific
 * order of this step is undefined.
 *
 * In all cases above, if the implementation does not support all the specified
 * features (reqtypes), it will be skipped and processing will continue from
 * where it left off. This means that if verto_new() returns non-NULL it is
 * guaranteed to support the features you specified.
 *
 * @see verto_set_default()
 * @param impl The implementation to use, or NULL.
 * @param reqtypes A bitwise or'd list of required event type features.
 * @return A new verto_ctx, or NULL on error.  Call verto_free() when done.
 */
verto_ctx *
verto_new(const char *impl, verto_ev_type reqtypes);

/**
 * Gets the default event context using an optionally specified implementation.
 *
 * This function is essentially a singleton version of verto_new().  However,
 * since this function must return the same loop as the *_default() call of
 * the underlying implementation (if such a function exists), it is NOT a
 * global singleton, but a per-implementation singleton. For this reason, you
 * must call verto_free() when you are done with this loop. Even after calling
 * verto_free() on the default verto_ctx, you can safely call verto_default()
 * again and receive a new reference to the same (internally default) loop.
 *
 * In all other respects, verto_default() acts exactly like verto_new().
 *
 * @see verto_new()
 * @see verto_free()
 * @param impl The implementation to use, or NULL.
 * @param reqtypes A bitwise or'd list of required event type features.
 * @return The default verto_ctx, or NULL on error.  Call verto_free() when done.
 */
verto_ctx *
verto_default(const char *impl, verto_ev_type reqtypes);

/**
 * Sets the default implementation to use by its name.
 *
 * This function returns 1 on success and 0 on failure.  It can fail for the
 * following reasons:
 *   1. The default implementation was already set via verto_set_default().
 *   2. The implementation specified could not be found.
 *   3. The implementation specified didn't support the features specified.
 *   4. The impl argument was NULL.
 *   5. verto_new() was already called.
 *   6. verto_default() was already called.
 *   7. verto_new_NAME() was already called.
 *   8. verto_default_NAME() was already called.
 *   9. verto_convert_NAME() was already called.
 *
 * @see verto_new()
 * @see verto_default()
 * @param impl The implementation to use.
 * @param reqtypes A bitwise or'd list of required event type features.
 * @return The default verto_ctx, or NULL on error.  Call verto_free() when done.
 */
int
verto_set_default(const char *impl, verto_ev_type reqtypes);

/**
 * Sets the allocator to use for verto_ctx and verto_ev objects.
 *
 * If you plan to set the allocator, you MUST call this function before any
 * other verto_*() calls.
 *
 * @see verto_new()
 * @see verto_default()
 * @see verto_add_io()
 * @see verto_add_timeout()
 * @see verto_add_idle()
 * @see verto_add_signal()
 * @see verto_add_child()
 * @param resize The allocator to use (behaves like realloc();
 *        resize(ptr, 0) must free memory at ptr.)
 * @param hierarchical Zero if the allocator is not hierarchical
 */
int
verto_set_allocator(void *(*resize)(void *mem, size_t size), int hierarchical);

/**
 * Frees a verto_ctx.
 *
 * When called on a default verto_ctx, the reference will be freed but the
 * internal default loop will still be available via another call to
 * verto_default().
 *
 * @see verto_new()
 * @see verto_default()
 * @param ctx The verto_ctx to free.
 */
void
verto_free(verto_ctx *ctx);

/**
 * Frees global state.
 *
 * Remove and free all allocated global state.  Call only when no further
 * contexts exist and all threads have exited.
 *
 * @see verto_new()
 * @see verto_free()
 * @see verto_default()
 */
void
verto_cleanup(void);

/**
 * Run the verto_ctx forever, or at least until verto_break() is called.
 *
 * @see verto_break()
 * @param ctx The verto_ctx to run.
 */
void
verto_run(verto_ctx *ctx);

/**
 * Run the verto_ctx once. May block.
 *
 * @param ctx The verto_ctx to run once.
 */
void
verto_run_once(verto_ctx *ctx);

/**
 * Exits the currently running verto_ctx.
 *
 * @see verto_run()
 * @param ctx The verto_ctx to exit.
 */
void
verto_break(verto_ctx *ctx);

/**
 * Re-initializes the verto_ctx.
 *
 * This function deletes all events, except those which have set the
 * VERTO_EV_FLAG_REINITIABLE flag. If you fork(), you MUST call this in the
 * child process after the fork!
 *
 * If this function fails it indicates that at least one
 * VERTO_EV_FLAG_REINITIABLE event was not rearmed or that ctx was NULL.
 *
 * @see verto_new()
 * @see verto_default()
 * @param ctx The verto_ctx to re-initialize.
 * @return Non-zero on success, 0 on error.
 */
int
verto_reinitialize(verto_ctx *ctx);

/**
 * Adds a callback executed when a file descriptor is ready to be read/written.
 *
 * All verto_ev events are automatically freed when their parent verto_ctx is
 * freed. You do not need to free them manually. If VERTO_EV_FLAG_PERSIST is
 * provided, the event will repeat until verto_del() is called. If
 * VERTO_EV_FLAG_PERSIST is not provided, the event will be freed automatically
 * after its execution. In either case, you may call verto_del() at any time
 * to prevent the event from executing.
 * If VERTO_EV_FLAG_IO_CLOSE_FD is provided the passed in fd is automatically
 * closed when the event is freed with verto_del()
 *
 * NOTE: On Windows, the underlying select() only works with sockets. As such,
 * any attempt to add a non-socket io event on Windows will produce undefined
 * results and may even crash.
 *
 * @see verto_del()
 * @param ctx The verto_ctx which will fire the callback.
 * @param flags The flags to set (at least one VERTO_EV_FLAG_IO* required).
 * @param callback The callback to fire.
 * @param fd The file descriptor to watch for reads.
 * @return The verto_ev registered with the event context or NULL on error.
 */
verto_ev *
verto_add_io(verto_ctx *ctx, verto_ev_flag flags,
             verto_callback *callback, int fd);

/**
 * Adds a callback executed after a period of time.
 *
 * All verto_ev events are automatically freed when their parent verto_ctx is
 * freed. You do not need to free them manually. If VERTO_EV_FLAG_PERSIST is
 * provided, the event will repeat until verto_del() is called. If
 * VERTO_EV_FLAG_PERSIST is not provided, the event will be freed automatically
 * after its execution. In either case, you may call verto_del() at any time
 * to prevent the event from executing.
 *
 * @see verto_del()
 * @param ctx The verto_ctx which will fire the callback.
 * @param flags The flags to set.
 * @param callback The callback to fire.
 * @param interval Time period to wait before firing (in milliseconds).
 * @return The verto_ev registered with the event context.
 */
verto_ev *
verto_add_timeout(verto_ctx *ctx, verto_ev_flag flags,
                  verto_callback *callback, time_t interval);

/**
 * Adds a callback executed when there is nothing else to do.
 *
 * All verto_ev events are automatically freed when their parent verto_ctx is
 * freed. You do not need to free them manually. If VERTO_EV_FLAG_PERSIST is
 * provided, the event will repeat until verto_del() is called. If
 * VERTO_EV_FLAG_PERSIST is not provided, the event will be freed automatically
 * after its execution. In either case, you may call verto_del() at any time
 * to prevent the event from executing.
 *
 * @see verto_del()
 * @param ctx The verto_ctx which will fire the callback.
 * @param flags The flags to set.
 * @param callback The callback to fire.
 * @return The verto_ev registered with the event context.
 */
verto_ev *
verto_add_idle(verto_ctx *ctx, verto_ev_flag flags,
               verto_callback *callback);

/**
 * Adds a callback executed when a signal is received.
 *
 * All verto_ev events are automatically freed when their parent verto_ctx is
 * freed. You do not need to free them manually. If VERTO_EV_FLAG_PERSIST is
 * provided, the event will repeat until verto_del() is called. If
 * VERTO_EV_FLAG_PERSIST is not provided, the event will be freed automatically
 * after its execution. In either case, you may call verto_del() at any time
 * to prevent the event from executing.
 *
 * NOTE: If you attempt to ignore a signal without the VERTO_EV_FLAG_PERSIST
 * flag, this function fails.
 *
 * NOTE: SIGCHLD is expressly not supported. If you want this notification,
 * please use verto_add_child().
 *
 * WARNNIG: Signal events can only be reliably received in the default verto_ctx
 * in some implementations.  Attempting to receive signal events in non-default
 * loops may result in assert() failures.
 *
 * WARNING: While verto does its best to protect you from crashes, there is
 * essentially no way to do signal events if you mix multiple implementations in
 * a single process. Attempting to do so will result in undefined behavior,
 * and potentially even a crash. You have been warned.
 *
 * @see verto_add_child()
 * @see verto_repeat()
 * @see verto_del()
 * @param ctx The verto_ctx which will fire the callback.
 * @param flags The flags to set.
 * @param callback The callback to fire.
 * @param signal The signal to watch for.
 * @return The verto_ev registered with the event context.
 */
verto_ev *
verto_add_signal(verto_ctx *ctx, verto_ev_flag flags,
                 verto_callback *callback, int signal);

/**
 * Adds a callback executed when a child process exits.
 *
 * This event will be freed automatically after its execution. Due to the
 * nature of a process' life-cycle, child events cannot persist (processes only
 * exit once). This function returns NULL if you attempt to use
 * VERTO_EV_FLAG_PERSIST. You may, of course, call verto_del() at any time to
 * prevent the callback from firing.
 *
 * @see verto_del()
 * @param ctx The verto_ctx which will fire the callback.
 * @param flags The flags to set.
 * @param callback The callback to fire.
 * @param child The pid (POSIX) or handle (Win32) of the child to watch for.
 * @return The verto_ev registered with the event context.
 */
verto_ev *
verto_add_child(verto_ctx *ctx, verto_ev_flag flags,
                verto_callback *callback, verto_proc proc);

/**
 * Sets the private pointer of the verto_ev.
 *
 * The free callback will be called in two cases:
 *   1. When the event is deleted (manually or automatically)
 *   2. When verto_set_private() is called again, unless
 *      free is NULL.
 *
 * @see verto_get_private()
 * @param ev The verto_ev
 * @param priv The private value to store
 * @param free The callback used to free the data or NULL
 */
void
verto_set_private(verto_ev *ev, void *priv, verto_callback *free);

/**
 * Gets the private pointer of the verto_ev.
 *
 * @see verto_set_private()
 * @param ev The verto_ev
 * @return The verto_ev private pointer
 */
void *
verto_get_private(const verto_ev *ev);

/**
 * Gets the type of the verto_ev.
 *
 * @see verto_add_io()
 * @see verto_add_timeout()
 * @see verto_add_idle()
 * @see verto_add_signal()
 * @see verto_add_child()
 * @param ev The verto_ev
 * @return The verto_ev type
 */
verto_ev_type
verto_get_type(const verto_ev *ev);

/**
 * Gets the flags associated with the given verto_ev.
 *
 * @see verto_add_io()
 * @see verto_add_timeout()
 * @see verto_add_idle()
 * @see verto_add_signal()
 * @see verto_add_child()
 * @see verto_set_flags()
 * @param ev The verto_ev
 * @return The verto_ev type
 */
verto_ev_flag
verto_get_flags(const verto_ev *ev);

/**
 * Sets the flags associated with the given verto_ev.
 *
 * See _VERTO_EV_FLAG_MUTABLE_MASK for the flags that can be changed
 * with this function. All others will be ignored. If the flags specified
 * are the same as the flags the event already has, this function is a no-op.
 *
 * @see verto_add_io()
 * @see verto_add_timeout()
 * @see verto_add_idle()
 * @see verto_add_signal()
 * @see verto_add_child()
 * @see verto_get_flags()
 * @param ev The verto_ev
 * @param flags The flags for the event
 */
void
verto_set_flags(verto_ev *ev, verto_ev_flag flags);

/**
 * Gets the file descriptor associated with a read/write verto_ev.
 *
 * @see verto_add_io()
 * @param ev The verto_ev to retrieve the file descriptor from.
 * @return The file descriptor, or -1 if not a read/write event.
 */
int
verto_get_fd(const verto_ev *ev);

/**
 * Gets the file descriptor state from when the event fires.
 *
 * @see verto_add_io()
 * @param ev The verto_ev to retrieve the fd state from.
 * @return The fd state.
 */
verto_ev_flag
verto_get_fd_state(const verto_ev *ev);

/**
 * Gets the interval associated with a timeout verto_ev.
 *
 * @see verto_add_timeout()
 * @param ev The verto_ev to retrieve the interval from.
 * @return The interval, or 0 if not a timeout event.
 */
time_t
verto_get_interval(const verto_ev *ev);

/**
 * Gets the signal associated with a signal verto_ev.
 *
 * @see verto_add_signal()
 * @param ev The verto_ev to retrieve the signal from.
 * @return The signal, or -1 if not a signal event.
 */
int
verto_get_signal(const verto_ev *ev);

/**
 * Gets the process associated with a child verto_ev.
 *
 * @see verto_add_child()
 * @param ev The verto_ev to retrieve the process from.
 * @return The pid/handle, or 0/NULL if not a child event (POSIX/Win32).
 */
verto_proc
verto_get_proc(const verto_ev *ev);

/**
 * Gets the status of the process which caused this event to fire.
 *
 * @see verto_add_child()
 * @param ev The verto_ev to retrieve the status from.
 * @return The pid/handle status.
 */
verto_proc_status
verto_get_proc_status(const verto_ev *ev);

/**
 * Gets the verto_ctx associated with a verto_ev.
 *
 * This is a borrowed reference, don't attempt to free it!
 *
 * @param ev The verto_ev to retrieve the verto_ctx from.
 * @return The verto_ctx.
 */
verto_ctx *
verto_get_ctx(const verto_ev *ev);

/**
 * Removes an event from from the event context and frees it.
 *
 * The event and its contents cannot be used after this call.
 *
 * @see verto_add_io()
 * @see verto_add_timeout()
 * @see verto_add_idle()
 * @see verto_add_signal()
 * @see verto_add_child()
 * @param ev The event to delete.
 */
void
verto_del(verto_ev *ev);

/**
 * Returns the event types supported by this implementation.
 *
 * @param ctx The verto_ctx to query.
 * @return The event types supported.
 */
verto_ev_type
verto_get_supported_types(verto_ctx *ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
#endif /* VERTO_H_ */
