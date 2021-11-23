echo ========== DOWNLOADING ADL DATASET ==========
if [ -z $(ls ADL_DCASE_DATA.zip) ]
then
	curl -L -o ADL_DCASE_DATA.zip https://www.dropbox.com/s/est21b4tmvzymps/ADL_DCASE_DATA.zip?dl=1
fi
if [ -z $(which unzip) ]
then
	echo unzip not found, trying to install...
	sudo apt-get install unzip || echo install failed, please try and install manually
fi
unzip ADL_DCASE_DATA.zip -d ADL_DCASE_DATA
rm -v ADL_DCASE_DATA.zip
echo === DOWNLOAD FINISHED, FILES LOCATED IN ===
echo ./ADL_DCASE_DATA/{development,evaluation}

