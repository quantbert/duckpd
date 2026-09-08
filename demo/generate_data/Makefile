DATA_DIRECTORY ?= data
START_DATE ?= 2010-01-01
END_DATE ?= 2025-01-01
TICKER_START ?= 0
TICKER_COUNT ?= 100
SEED ?= 42
OVERWRITE ?= false
DRY_RUN ?= false
NEWS_SOURCE ?= source_data/dojo_stock_news.parquet
EMBEDDING_BATCH_SIZE ?= 1024
EMBEDDING_BACKEND ?= fastembed
EMBEDDING_DEVICE ?= cpu
TRANSFORMER_BATCH_SIZE ?= 64
NEWS_UV_RUN ?= uv run
NEWS_SOURCE_URL := https://huggingface.co/datasets/AlphaDojo/dojo_stock_news/resolve/9d7a641b5cb5eaedc1874c34e74aa186a0a768f1/data.parquet
NEWS_SOURCE_SHA256 := f8586e5a3515937da130b708006197652fd1560c2058d0dcd0134af5ebebe1fd

.DEFAULT_GOAL := help

OVERWRITE_FLAG := $(if $(filter true 1 yes,$(OVERWRITE)),--overwrite,)
DRY_RUN_FLAG := $(if $(filter true 1 yes,$(DRY_RUN)),--dry-run,)

.PHONY: help data generate download-news generate-news upload demo clean

help: ## Show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

data: generate generate-news upload ## Generate and upload the complete feature store

generate: ## Generate local sample data and metadata
	uv run --group generation python gendata.py \
		--output "$(DATA_DIRECTORY)/ohlcv" \
		--start "$(START_DATE)" \
		--end "$(END_DATE)" \
		--ticker-start "$(TICKER_START)" \
		--ticker-count "$(TICKER_COUNT)" \
		--seed "$(SEED)" \
		--generate-sma \
		--sma-output "$(DATA_DIRECTORY)/sma" \
		--symbology-output "$(DATA_DIRECTORY)/symbols/data.parquet" \
		--markets-output "$(DATA_DIRECTORY)/markets/data.parquet" $(OVERWRITE_FLAG)

download-news: ## Download and verify the pinned news source
	mkdir -p "$(dir $(NEWS_SOURCE))"
	test -f "$(NEWS_SOURCE)" || curl --fail --location --retry 5 \
		--output "$(NEWS_SOURCE)" "$(NEWS_SOURCE_URL)"
	printf '%s  %s\n' "$(NEWS_SOURCE_SHA256)" "$(NEWS_SOURCE)" | sha256sum --check -

generate-news: download-news ## Generate the embedded news stress dataset
	$(NEWS_UV_RUN) --group generation --extra embeddings python gendata.py \
		--news-only \
		--news-source "$(NEWS_SOURCE)" \
		--news-output "$(DATA_DIRECTORY)/news" \
		--start "$(START_DATE)" \
		--end "$(END_DATE)" \
		--ticker-start "$(TICKER_START)" \
		--ticker-count "$(TICKER_COUNT)" \
		--embedding-batch-size "$(EMBEDDING_BATCH_SIZE)" \
		--embedding-backend "$(EMBEDDING_BACKEND)" \
		--embedding-device "$(EMBEDDING_DEVICE)" \
		--transformer-batch-size "$(TRANSFORMER_BATCH_SIZE)" $(OVERWRITE_FLAG)

upload: ## Upload the generated feature store to Hugging Face
	uv run --extra featurestore --env-file .env python hfupload.py \
		--data "$(DATA_DIRECTORY)" $(DRY_RUN_FLAG)

demo: ## Run the remote cache and query example
	uv run --extra featurestore --env-file .env python demo.py

clean: ## Remove generated data and the feature cache
	rm -rf "$(DATA_DIRECTORY)" feature_cache