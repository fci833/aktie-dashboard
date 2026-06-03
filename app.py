# Slet den forkert placerede config.toml
rm -f config.toml

# Lav den korrekte mappe og fil
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[theme]
base = "dark"
primaryColor = "#00d4aa"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1a1f2e"
textColor = "#fafafa"
font = "sans serif"

[server]
maxUploadSize = 50
EOF

# Skriv requirements.txt korrekt
cat > requirements.txt << 'EOF'
streamlit==1.39.0
yfinance==0.2.50
pandas==2.2.3
numpy==1.26.4
plotly==5.24.1
ta==0.11.0
scipy==1.14.1
requests==2.32.3
EOF

echo "✅ requirements.txt og config.toml er rettet!"
ls -la
