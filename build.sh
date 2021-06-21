#!/bin/bash

echo '1/4 Installing yarn dependencies'
yarn add gatsby react react-dom
yarn add @fortawesome/fontawesome-svg-core
yarn add @fortawesome/free-solid-svg-icons
yarn add @fortawesome/react-fontawesome

yarn add https://github.com/pbamotra/gatsby-theme-andy#pbamotrav2

echo '2/4 Yarn build'
yarn clean && yarn build

echo '3/4 Jekyll build'
jekyll build

echo '4/4 Finished complete build'