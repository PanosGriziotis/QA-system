#!/bin/sh
set -e

# wiki dump name 
WIKI_DUMP_NAME=elwiki-20240501-pages-articles-multistream.xml.bz2
WIKI_DUMP_DOWNLOAD_URL=https://mirror.accum.se/mirror/wikimedia.org/dumps/elwiki/20240501/$WIKI_DUMP_NAME

echo "Downloading el-language Wikipedia dump from $WIKI_DUMP_DOWNLOAD_URL..."
wget -c $WIKI_DUMP_DOWNLOAD_URL
echo "Succesfully downloaded the latest el-language Wikipedia dump to $WIKI_DUMP_NAME"
