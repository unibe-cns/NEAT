#!/bin/sh
# This is a simple script to renew the sphinx documentation from the source code

make clean 								# clean all build folders
sphinx-apidoc -o source/ ../neat		# renew all documentation from python api
make html								# renew all html code