#ifndef SRC_LLM
#define SRC_LLM
#include <vector>
#include <sstream>
#include <llama.h>
#include <common/common.h>

struct ChatMessage {
    std::string role;
    std::string content;
};

class LLM {
private:
    llama_context* ctx;
    llama_model* model;
    llama_batch batch;
    std::vector<llama_token> tokensList;
    std::vector<ChatMessage> messages;
    std::vector<std::string> errors;
    const int batchSize = 512;
    int nCtx;

    void tokenizeChatML() {
        tokensList = {};
        for (auto message : messages) {
            tokensList.push_back(llama_token_bos(model));
            auto tokens = ::llama_tokenize(ctx, message.role + '\n' + message.content, true);
            tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
            tokensList.push_back(llama_token_eos(model));
        }

        tokensList.push_back(llama_token_bos(model));
        auto tokens = ::llama_tokenize(ctx, "assistant\n", true);
        tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
    }

    void decodeTokens(std::vector<llama_token> tokens) {
        int batchesNeeded = std::ceil(float(tokens.size()) / float(batchSize));
        for (int i = 0; i < batchesNeeded; i++) {
            llama_batch_clear(batch);
            int jUpper = (tokens.size()-i*batchSize < batchSize) ? tokens.size()-i*batchSize : batchSize;
            for (size_t j = 0; j < jUpper; j++) {
                llama_batch_add(batch, tokens[j+i*batchSize], j+i*batchSize, { 0 }, false);
            }

            if (i == batchesNeeded-1) batch.logits[batch.n_tokens - 1] = true;
            llama_decode(ctx, batch);
        }
    }

    void decodeMessage(std::string message) {
        tokensList = ::llama_tokenize(ctx, message, true);
        decodeTokens(tokensList);
    }

public:
    LLM(std::string modelPath, int seed, int nCtx, int nThreads, int nThreadsBatch, std::string systemMessage = "", int nGpuLayers = -1, bool numa = false) {
        this->nCtx = nCtx;
        llama_backend_init(numa);

        llama_model_params modelParams = llama_model_default_params();
        modelParams.n_gpu_layers = nGpuLayers;

        model = llama_load_model_from_file(modelPath.c_str(), modelParams);

        if (model == nullptr)
            errors.push_back("Unable to load model at path " + modelPath);

        llama_context_params ctxParams = llama_context_default_params();

        ctxParams.seed = seed;
        ctxParams.n_ctx = nCtx;
        ctxParams.n_threads = nThreads;

        ctxParams.n_threads_batch = nThreadsBatch == -1 ? nThreads : nThreadsBatch;

        ctx = llama_new_context_with_model(model, ctxParams);

        if (ctx == nullptr)
            errors.push_back("Unable to initialize context");

        batch = llama_batch_init(batchSize, 0, 1);

        const auto t_main_start = ggml_time_us();

        if (systemMessage != "") {
            messages.push_back(ChatMessage{ "system", systemMessage });
        }
    }

    ~LLM() {
        llama_batch_free(batch);

        llama_free(ctx);
        llama_free_model(model);
    }

    std::vector<std::string> getErrors() {
        return errors;
    }

    std::string response(std::string prompt, int nLen, bool live = false) {
        llama_reset_timings(ctx);
        messages.push_back(ChatMessage{ "user", prompt });

        if (nLen > nCtx)
            return "Error: nLen > nCtx";

        llama_kv_cache_clear(ctx);

        tokenizeChatML();
        decodeTokens(tokensList);

        std::stringstream response;
        int nCur = tokensList.size();
        bool tagFound = false;
        std::string tag;

        while (nCur < nLen) {
            auto nVocab = llama_n_vocab(model);
            auto* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(nVocab);

            for (llama_token tokenId = 0; tokenId < nVocab; tokenId++)
                candidates.emplace_back(llama_token_data{ tokenId, logits[tokenId], 0.0f });

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            const llama_token newTokenId = llama_sample_token(ctx, &candidates_p);

            if (newTokenId == llama_token_eos(model) || nCur == nLen) {
                llama_batch_clear(batch);
                llama_batch_add(batch, newTokenId, nCur, { 0 }, true);
                llama_decode(ctx, batch);

                nCur += 1;
                break;
            }
            
            response << llama_token_to_piece(ctx, newTokenId);

            if (live) {
                std::cout << llama_token_to_piece(ctx, newTokenId);
                std::fflush(stdout);
            }

            llama_batch_clear(batch);
            llama_batch_add(batch, newTokenId, nCur, { 0 }, true);
            if (llama_decode(ctx, batch))
                return "Error: llama_decode() failed";

            nCur += 1;
        }

        messages.push_back(ChatMessage{ "assistant", response.str() });

        return messages.back().content;
    }

    void printTimings() {
        llama_print_timings(ctx);
    }

    bool ok() {
        return errors.size() == 0;
    }
};

#endif /* SRC_LLM */
