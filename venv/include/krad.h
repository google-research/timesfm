/* -*- mode: c; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/*
 * Copyright 2013 Red Hat, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This API is not considered as stable as the main krb5 API.
 *
 * - We may make arbitrary incompatible changes between feature releases
 *   (e.g. from 1.12 to 1.13).
 * - We will make some effort to avoid making incompatible changes for
 *   bugfix releases, but will make them if necessary.
 */

#ifndef KRAD_H_
#define KRAD_H_

#include <krb5.h>
#include <verto.h>
#include <stddef.h>
#include <stdio.h>

#define KRAD_PACKET_SIZE_MAX 4096

#define KRAD_SERVICE_TYPE_LOGIN 1
#define KRAD_SERVICE_TYPE_FRAMED 2
#define KRAD_SERVICE_TYPE_CALLBACK_LOGIN 3
#define KRAD_SERVICE_TYPE_CALLBACK_FRAMED 4
#define KRAD_SERVICE_TYPE_OUTBOUND 5
#define KRAD_SERVICE_TYPE_ADMINISTRATIVE 6
#define KRAD_SERVICE_TYPE_NAS_PROMPT 7
#define KRAD_SERVICE_TYPE_AUTHENTICATE_ONLY 8
#define KRAD_SERVICE_TYPE_CALLBACK_NAS_PROMPT 9
#define KRAD_SERVICE_TYPE_CALL_CHECK 10
#define KRAD_SERVICE_TYPE_CALLBACK_ADMINISTRATIVE 11

typedef struct krad_attrset_st krad_attrset;
typedef struct krad_packet_st krad_packet;
typedef struct krad_client_st krad_client;
typedef unsigned char krad_code;
typedef unsigned char krad_attr;

/* Called when a response is received or the request times out. */
typedef void
(*krad_cb)(krb5_error_code retval, const krad_packet *request,
           const krad_packet *response, void *data);

/*
 * Called to iterate over a set of requests.  Either the callback will be
 * called until it returns NULL, or it will be called with cancel = TRUE to
 * terminate in the middle of an iteration.
 */
typedef const krad_packet *
(*krad_packet_iter_cb)(void *data, krb5_boolean cancel);

/*
 * Code
 */

/* Convert a code name to its number. Only works for codes defined
 * by RFC 2875 or 2882. Returns 0 if the name was not found. */
krad_code
krad_code_name2num(const char *name);

/* Convert a code number to its name. Only works for attributes defined
 * by RFC 2865 or 2882. Returns NULL if the name was not found. */
const char *
krad_code_num2name(krad_code code);

/*
 * Attribute
 */

/* Convert an attribute name to its number. Only works for attributes defined
 * by RFC 2865. Returns 0 if the name was not found. */
krad_attr
krad_attr_name2num(const char *name);

/* Convert an attribute number to its name. Only works for attributes defined
 * by RFC 2865. Returns NULL if the name was not found. */
const char *
krad_attr_num2name(krad_attr type);

/*
 * Attribute set
 */

/* Create a new attribute set. */
krb5_error_code
krad_attrset_new(krb5_context ctx, krad_attrset **set);

/* Create a deep copy of an attribute set. */
krb5_error_code
krad_attrset_copy(const krad_attrset *set, krad_attrset **copy);

/* Free an attribute set. */
void
krad_attrset_free(krad_attrset *set);

/* Add an attribute to a set. */
krb5_error_code
krad_attrset_add(krad_attrset *set, krad_attr type, const krb5_data *data);

/* Add a four-octet unsigned number attribute to the given set. */
krb5_error_code
krad_attrset_add_number(krad_attrset *set, krad_attr type, krb5_ui_4 num);

/* Delete the specified attribute. */
void
krad_attrset_del(krad_attrset *set, krad_attr type, size_t indx);

/* Get the specified attribute. */
const krb5_data *
krad_attrset_get(const krad_attrset *set, krad_attr type, size_t indx);

/*
 * Packet
 */

/* Determine the bytes needed from the socket to get the whole packet.  Don't
 * cache the return value as it can change! Returns -1 on EBADMSG. */
ssize_t
krad_packet_bytes_needed(const krb5_data *buffer);

/* Free a packet. */
void
krad_packet_free(krad_packet *pkt);

/*
 * Create a new request packet.
 *
 * This function takes the attributes specified in set and converts them into a
 * radius packet. The packet will have a randomized id. If cb is not NULL, it
 * will be called passing data as the argument to iterate over a set of
 * outstanding requests. In this case, the id will be both random and unique
 * across the set of requests.
 */
krb5_error_code
krad_packet_new_request(krb5_context ctx, const char *secret, krad_code code,
                        const krad_attrset *set, krad_packet_iter_cb cb,
                        void *data, krad_packet **request);

/*
 * Create a new response packet.
 *
 * This function is similar to krad_packet_new_requst() except that it crafts a
 * packet in response to a request packet. This new packet will borrow values
 * from the request such as the id and the authenticator.
 */
krb5_error_code
krad_packet_new_response(krb5_context ctx, const char *secret, krad_code code,
                         const krad_attrset *set, const krad_packet *request,
                         krad_packet **response);

/*
 * Decode a request radius packet from krb5_data.
 *
 * The resulting decoded packet will be a request packet stored in *reqpkt.
 *
 * If cb is NULL, *duppkt will always be NULL.
 *
 * If cb is not NULL, it will be called (with the data argument) to iterate
 * over a set of requests currently being processed. In this case, if the
 * packet is a duplicate of an already received request, the original request
 * will be set in *duppkt.
 */
krb5_error_code
krad_packet_decode_request(krb5_context ctx, const char *secret,
                           const krb5_data *buffer, krad_packet_iter_cb cb,
                           void *data, const krad_packet **duppkt,
                           krad_packet **reqpkt);

/*
 * Decode a response radius packet from krb5_data.
 *
 * The resulting decoded packet will be a response packet stored in *rsppkt.
 *
 * If cb is NULL, *reqpkt will always be NULL.
 *
 * If cb is not NULL, it will be called (with the data argument) to iterate
 * over a set of requests awaiting responses. In this case, if the response
 * packet matches one of these requests, the original request will be set in
 * *reqpkt.
 */
krb5_error_code
krad_packet_decode_response(krb5_context ctx, const char *secret,
                            const krb5_data *buffer, krad_packet_iter_cb cb,
                            void *data, const krad_packet **reqpkt,
                            krad_packet **rsppkt);

/* Encode packet. */
const krb5_data *
krad_packet_encode(const krad_packet *pkt);

/* Get the code for the given packet. */
krad_code
krad_packet_get_code(const krad_packet *pkt);

/* Get the specified attribute. */
const krb5_data *
krad_packet_get_attr(const krad_packet *pkt, krad_attr type, size_t indx);

/*
 * Client
 */

/* Create a new client. */
krb5_error_code
krad_client_new(krb5_context kctx, verto_ctx *vctx, krad_client **client);

/* Free the client. */
void
krad_client_free(krad_client *client);

/*
 * Send a request to a radius server.
 *
 * The remote host may be specified by one of the following formats:
 *  - /path/to/unix.socket
 *  - IPv4
 *  - IPv4:port
 *  - IPv4:service
 *  - [IPv6]
 *  - [IPv6]:port
 *  - [IPv6]:service
 *  - hostname
 *  - hostname:port
 *  - hostname:service
 *
 * The timeout parameter (milliseconds) is the total timeout across all remote
 * hosts (when DNS returns multiple entries) and all retries.  For stream
 * sockets, the retries parameter is ignored and no retries are performed.
 *
 * The cb function will be called with the data argument when either a response
 * is received or the request times out on all possible remote hosts.
 */
krb5_error_code
krad_client_send(krad_client *rc, krad_code code, const krad_attrset *attrs,
                 const char *remote, const char *secret, int timeout,
                 size_t retries, krad_cb cb, void *data);

#endif /* KRAD_H_ */
