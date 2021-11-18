download() {
	wget "$1" -O "$2"
}
download "https://example.com" ADL_DCASE_DATA.zip
unzip ADL_DCASE_DATA.zip
rm -v ADL_DCASE_DATA.zip
