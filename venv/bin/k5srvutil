#!/bin/sh

# list_princs keytab
# returns a list of principals in the keytab
# sorted and uniquified
list_princs() {
    klist -k $keytab | awk '(NR > 3) {print $2}' | sort | uniq
}

set_command() {
    if [ x$command != x ] ; then
	cmd_error Only one command can be specified
	usage
	exit 1
    fi
    command=$1
}

#interactive_prompt prompt princ
# If in interactive mode  return  true if the principal  should be acted on
# otherwise return true all the time
interactive_prompt() {
    if [ $interactive = 0 ] ; then
	return 0
    fi
    printf "%s for %s? [yn]" "$1" "$2"
    read ans
    case $ans in
    n*|N*)
	return 1
	;;
    esac
    return 0
    }
    
cmd_error() {
    echo $@ 2>&1
    }

usage() {
    echo "Usage: $0 [-i] [-f file] [-e keysalts] list|change|delete|delold"
}



change_key() {
    princs=`list_princs `
    for princ in $princs; do
	if interactive_prompt "Change key " $princ; then
	    kadmin -k -t $keytab -p $princ -q \
		"ktadd -k $keytab $keysalts $princ"
	fi
    done
    }

delete_old_keys() {
    princs=`list_princs `
    for princ in $princs; do
	if interactive_prompt "Delete old keys " $princ; then
	    kadmin -k -t $keytab -p $princ -q "ktrem -k $keytab $princ old"
	fi
    done
    }

delete_keys() {
    interactive=1
    princs=`list_princs `
    for princ in $princs; do
	if interactive_prompt "Delete all keys " $princ; then
	    kadmin -p $princ -k -t $keytab -q "ktrem -k $keytab $princ all"
	fi
    done
    }


keytab=/etc/krb5.keytab
interactive=0
keysalts=""

while [ $# -gt 0 ] ; do
    opt=$1
    shift
        case $opt in
	"-f")
	keytab=$1
	shift
	;;
	"-i")
	interactive=1
	;;
	"-e")
	keysalts="$keysalts -e \"$1\""
	shift
	;;
	change|delold|delete|list)
	set_command $opt
	;;
	*)
	cmd_error Illegal option: $opt
	usage
	exit 1
	;;
	esac
done
	

case $command in
    change)
    change_key
    ;;
    delold)
    delete_old_keys
    ;;
    delete)
    delete_keys
    ;;
    list)
    klist -k $keytab
    ;;
    *)
        usage
	;;
    esac
