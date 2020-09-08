#!/usr/bin/env bash
#
# This is an adaptation of code I wrote to download a private binary from GitHub. Such...pain.
# Why can't GitHub just offer a standardized URL you can download a release binary from and attach 
# your Github Personal Access Token as a header? 
#
# Since this code is an adaptation it hasn't been directly tested, but the code it was adapted from works
# and hopefully you can get the missing piece you're after by looking here.
#

set -e

# Parse CLI args
readonly github_oauth_token="$1"
readonly git_tag="$2"
readonly github_repo_owner="$3"
readonly github_repo_name="$4"
readonly release_asset_filename="$5"
readonly output_path="$6"

# Get the "github tag id" of this release
github_tag_id=$(curl --silent --show-error \
                     --header "Authorization: token $github_oauth_token" \
                     --request GET \
                     "https://api.github.com/repos/$github_repo_owner/$github_repo_name/releases" \
                     | jq --raw-output ".[] | select(.tag_name==\"$git_tag\").id")

# Get the download URL of our desired asset
download_url=$(curl --silent --show-error \
                    --header "Authorization: token $github_oauth_token" \
                    --header "Accept: application/vnd.github.v3.raw" \
                    --location \
                    --request GET \
                    "https://api.github.com/repos/$github_repo_owner/$github_repo_name/releases/$github_tag_id" \
                    | jq --raw-output ".assets[] | select(.name==\"$release_asset_filename\").url")

# Get GitHub's S3 redirect URL
# Why not just curl's built-in "--location" option to auto-redirect? Because curl then wants to include all the original
# headers we added for the GitHub request, which makes AWS complain that we're trying strange things to authenticate.
redirect_url=$(curl --silent --show-error \
          --header "Authorization: token $github_oauth_token" \
          --header "Accept: application/octet-stream" \
          --request GET \
          --write-out "%{redirect_url}" \
          "$download_url")

# Finally download the actual binary
curl --silent --show-error \
          --header "Accept: application/octet-stream" \
          --output "$output_path" \
          --request GET \
          "$redirect_url"