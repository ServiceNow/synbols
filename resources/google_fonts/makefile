GOOGLE_FONTS_COMMIT=ed61614fb47affd2a4ef286e0b313c5c47226c69
download:
	wget -O google_fonts.zip https://github.com/google/fonts/archive/${GOOGLE_FONTS_COMMIT}.zip
	unzip google_fonts.zip
	mv fonts-${GOOGLE_FONTS_COMMIT} google_fonts

add-synbols-prefix:
	find google_fonts -name "*.ttf" | xargs -P4 -I{} sh -c "python3 rename_fonts.py {}" 2> rename.err

PACKAGE_DIRNAME = synbols_google_fonts
ARCHIVE_NAME = ${PACKAGE_DIRNAME}.zip
TO_DELETE =  ${PACKAGE_DIRNAME}/.ci
TO_DELETE := ${TO_DELETE} ${PACKAGE_DIRNAME}/tools
TO_DELETE := ${TO_DELETE} ${PACKAGE_DIRNAME}/.gitattributes synbols_google_fonts/.gitignore synbols_google_fonts/.gitmodules
TO_DELETE := ${TO_DELETE} ${PACKAGE_DIRNAME}/.travis.yml
TO_DELETE := ${TO_DELETE} ${PACKAGE_DIRNAME}/CONTRIBUTING.md
TO_DELETE := ${TO_DELETE} ${PACKAGE_DIRNAME}/to_production.txt
TO_DELETE := ${TO_DELETE} ${PACKAGE_DIRNAME}/to_sandbox.txt
package:
	cp -r google_fonts ${PACKAGE_DIRNAME}
	find ${PACKAGE_DIRNAME} -name "*.ttf" | grep -v "synbols-" | xargs -I{} rm {}
	rm -r ${TO_DELETE}
	zip -r ${ARCHIVE_NAME} ${PACKAGE_DIRNAME}

clean:
	rm -r google_fonts google_fonts.zip ${PACKAGE_DIRNAME} rename.log rename.err

all: download add-synbols-prefix package clean

