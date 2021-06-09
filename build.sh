#!/bin/bash

echo '1/4 Installing yarn dependencies'
yarn add gatsby react react-dom
yarn add https://github.com/pbamotra/gatsby-theme-andy#pbamotra

echo '2/4 Yarn build'
yarn clean && yarn build

echo '3/4 Jekyll build'
jekyll build

echo '4/4 Finished complete build'