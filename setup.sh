#mkdir -p ~/.streamlit/
#
#echo "\
#[general]\n\
#email = \"your-email@domain.com\"\n\
#" > ~/.streamlit/credentials.toml
#
#echo "\
#[server]\n\
#headless = true\n\
#enableCORS=false\n\
#port = $PORT\n\
mkdir -p ~/.streamlit

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml

