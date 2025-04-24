install:
	pip install -r requirements.txt

app:
	streamlit run src/main.py

deploy:
	install app