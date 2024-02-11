python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run "C:/Users/Мария/Desktop/streamlit_etna/streamlit_app.py" [ARGUMENTS]

Result:
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.1.102:8501

If you get error AttributeError: module 'lib' has no attribute 'X509_V_FLAG_CB_ISSUER_CHECK'
rm  c:\anaconda\lib\site-packages\OpenSSL\
pip3 install pyopenssl
pip3 install pyopenssl --upgrade
