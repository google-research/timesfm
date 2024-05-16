#!/bin/sh
###############################################################################
#
# Copyright (C) 2005 Red Hat, Inc. All Rights Reserved.
# Written by David Howells (dhowells@redhat.com)
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version
# 2 of the License, or (at your option) any later version.
#
###############################################################################
#
# Request key debugging
#
# Call: request-key-debug.sh <keyid> <desc> <callout> <session-keyring>
#

echo RQDebug keyid: $1
echo RQDebug desc: $2
echo RQDebug callout: $3
echo RQDebug session keyring: $4

if [ "$3" != "neg" ]
then
    keyctl instantiate $1 "Debug $3" $4 || exit 1
else
    cat /proc/keys
    echo keyctl negate $1 30 $4
    keyctl negate $1 30 $4
fi

exit 0
