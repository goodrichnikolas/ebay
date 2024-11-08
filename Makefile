deploy_ollama:
	@echo "Deploying Ollama"
	@echo "Deploying Ollama to Heroku"
	ollama list
	ollama serve