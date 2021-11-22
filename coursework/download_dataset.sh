echo ========== DOWNLOADING ADL DATASET ==========
curl -L -o ADL_DCASE_DATA.zip https://www.dropbox.com/s/est21b4tmvzymps/ADL_DCASE_DATA.zip?dl=1
unzip ADL_DCASE_DATA.zip -d ADL_DCASE_DATA
rm -v ADL_DCASE_DATA.zip
echo === DOWNLOAD FINISHED, FILES LOCATED IN ===
echo ./ADL_DCASE_DATA/{development,evaluation}
