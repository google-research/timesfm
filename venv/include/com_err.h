/*
 * Copyright 1988, Student Information Processing Board of the
 * Massachusetts Institute of Technology.
 *
 * Copyright 1995 by Cygnus Support.
 *
 * For copyright and distribution info, see the documentation supplied
 * with this package.
 */

/* Header file for common error description library. */

#ifndef __COM_ERR_H

#if defined(_WIN32)
#include <win-mac.h>
#endif

#ifndef KRB5_CALLCONV
#define KRB5_CALLCONV
#define KRB5_CALLCONV_C
#endif

#include <stdarg.h>

typedef long errcode_t;
typedef void (*et_old_error_hook_func) (const char *, errcode_t,
					const char *, va_list ap);

struct error_table {
	/*@shared@*/ char const * const * msgs;
        long base;
	unsigned int n_msgs;
};

#ifdef __cplusplus
extern "C" {
#endif

/* Public interfaces */
extern void KRB5_CALLCONV_C com_err
	(const char *, errcode_t, const char *, ...)
#if !defined(__cplusplus) && (__GNUC__ > 2)
    __attribute__((__format__(__printf__, 3, 4)))
#endif
    ;
extern void KRB5_CALLCONV com_err_va
	(const char *whoami, errcode_t code, const char *fmt,
	 va_list ap)
#if !defined(__cplusplus) && (__GNUC__ > 2)
    __attribute__((__format__(__printf__, 3, 0)))
#endif
    ;
extern /*@observer@*//*@dependent@*/ const char * KRB5_CALLCONV error_message
	(errcode_t)
       /*@modifies internalState@*/;
extern errcode_t KRB5_CALLCONV add_error_table
	(/*@dependent@*/ const struct error_table *)
       /*@modifies internalState@*/;
extern errcode_t KRB5_CALLCONV remove_error_table
	(const struct error_table *)
       /*@modifies internalState@*/;

#if !defined(_WIN32)
/*
 * The display routine should be application specific.  A global hook,
 * may cause inappropriate display procedures to be called between
 * applications under non-Unix environments.
 */

extern et_old_error_hook_func set_com_err_hook (et_old_error_hook_func);
extern et_old_error_hook_func reset_com_err_hook (void);
#endif

#ifdef __cplusplus
}
#endif

#define __COM_ERR_H
#endif /* ! defined(__COM_ERR_H) */
