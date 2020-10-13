build:
	python gentags.py
	bundle install && bundle update
	jekyll build
