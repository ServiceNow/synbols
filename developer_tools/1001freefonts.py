import os

from bs4 import BeautifulSoup
from collections import defaultdict
from robobrowser import RoboBrowser

BASE_URL = 'https://www.1001freefonts.com'
ACCEPTED_LICENSES = defaultdict(bool)
ACCEPTED_LICENSES['free'] = True
ACCEPTED_LICENSES['public_domain__gpl__ofl'] = True

br = RoboBrowser()

# Fonts are indexed by first letter
for prefix in 'abcdefghijklmnopqrstuvwxyz':
    print('%s fonts' % prefix.upper())
    
    # Iterate over all pages
    for page in range(1, 1000000):
        url = BASE_URL + '/%sfonts%d.php' % (prefix, page)
        br.open(url)
        
        soup = BeautifulSoup(br.response.content, features='html.parser')
        font_blocks = soup.find_all(class_='fontPreviewWrapper')
        
        # Check if we've reached the last page
        if len(font_blocks) == 0:
            break

        # Extract all fonts
        for tag in font_blocks:
            loc = tag.find(class_='downloadButtonElement').attrs['onclick'].split('=')[1][1:-2]
            font_name = os.path.basename(loc).replace('.zip', '')
            font_url = BASE_URL + loc
            license = tag.find(class_='previewLicenceTextCss').attrs['class'][1]
            if ACCEPTED_LICENSES[license]:
                print('...', font_name)
                if os.path.exists(font_name):
                    print('Already exists. Skip')
                    continue
                os.system('wget {0} 2> /dev/null && unzip -qq "{1}.zip" -d "{1}" 2> /dev/null && rm {1}.zip'.format(font_url, font_name))
                with open(os.path.join(font_name, 'extracted_license.txt'), 'w') as f:
                    f.write(license)
                    f.close()
