GOOGLE_FONTS_COMMIT=ed61614fb47affd2a4ef286e0b313c5c47226c69
download:
	wget -O google_fonts.zip https://github.com/google/fonts/archive/${GOOGLE_FONTS_COMMIT}.zip
	unzip google_fonts.zip
	mv fonts-${GOOGLE_FONTS_COMMIT} google_fonts

add-synbols-prefix:
	find google_fonts -name "METADATA.pb" | xargs -P4 -I{} sh -c "python3 rename_fonts.py {}" 2> rename.err

PACKAGE_DIRNAME = synbols_google_fonts
ARCHIVE_NAME = ${PACKAGE_DIRNAME}.zip
TO_DELETE =  ${PACKAGE_DIRNAME}/.ci
TO_DELETE := ${TO_DELETE} ${PACKAGE_DIRNAME}/tools
TO_DELETE := ${TO_DELETE} ${PACKAGE_DIRNAME}/.gitattributes ${PACKAGE_DIRNAME}/.gitignore ${PACKAGE_DIRNAME}/.gitmodules
TO_DELETE := ${TO_DELETE} ${PACKAGE_DIRNAME}/.travis.yml
TO_DELETE := ${TO_DELETE} ${PACKAGE_DIRNAME}/CONTRIBUTING.md
TO_DELETE := ${TO_DELETE} ${PACKAGE_DIRNAME}/to_production.txt
TO_DELETE := ${TO_DELETE} ${PACKAGE_DIRNAME}/to_sandbox.txt
package:
	# Gather all metadata
	find google_fonts -name "synbols_metadata" | xargs cat | uniq > google_fonts_metadata
	echo "synbols_name, original_name, license, designer, copyright" > google_fonts_licenses
	find google_fonts -name "synbols_licenses" | xargs cat | uniq >> google_fonts_licenses

	# Remove all non-synbols-renamed fonts
	cp -r google_fonts ${PACKAGE_DIRNAME}
	find ${PACKAGE_DIRNAME} -name "*.ttf" | grep -v "synbols-" | xargs -I{} rm {}

	# Clean package directory
	rm -r ${TO_DELETE}
	zip -r ${ARCHIVE_NAME} ${PACKAGE_DIRNAME}

clean:
	rm -r google_fonts google_fonts.zip ${PACKAGE_DIRNAME} rename.err

all: download add-synbols-prefix package clean

