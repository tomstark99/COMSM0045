<<<<<<< Updated upstream
download() {
	wget "$1" -O "$2"
}
download "https://example.com" ADL_DCASE_DATA.zip
unzip ADL_DCASE_DATA.zip
rm -v ADL_DCASE_DATA.zip
=======
echo ========== DOWNLOADING ADL DATASET ==========
curl -L -o ADL_DCASE_DATA.zip https://www.dropbox.com/s/est21b4tmvzymps/ADL_DCASE_DATA.zip?dl=1
unzip ADL_DCASE_DATA.zip -d ADL_DCASE_DATA
rm -v ADL_DCASE_DATA.zip
echo === DOWNLOAD FINISHED, FILES LOCATED IN ===
echo ./ADL_DCASE_DATA/{development,evaluation}
>>>>>>> Stashed changes
