/* SPDX-License-Identifier: MPL-2.0 */

/*  This file is deprecated, and all its functionality provided by zmq.h     */
/*  Note that -Wpedantic compilation requires GCC to avoid using its custom
    extensions such as #warning, hence the trick below. Also, pragmas for
    warnings or other messages are not standard, not portable, and not all
    compilers even have an equivalent concept.
    So in the worst case, this include file is treated as silently empty. */

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)               \
  || defined(_MSC_VER)
#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcpp"
#pragma GCC diagnostic ignored "-Werror"
#pragma GCC diagnostic ignored "-Wall"
#endif
#pragma message(                                                               \
  "Warning: zmq_utils.h is deprecated. All its functionality is provided by zmq.h.")
#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
#endif
