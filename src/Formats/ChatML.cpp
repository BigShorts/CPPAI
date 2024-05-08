#include "ChatML.hpp"

std::unordered_map<std::string, int> map = {
        { "<|begin_of_text|>", 128000 },
        { "<|end_of_text|>", 128001 },
        { "<|im_start|>", 128002 },
        { "<|im_end|>", 128003 },
        { "<tool_call>", 128004 },
        { "<tool_response>", 128005 },
        { "<|start_header_id|>", 128006 },
        { "<|end_header_id|>", 128007 },
        { "<|tools|>", 128008 },
        { "<|eot_id>", 128009 },
        { "</tools>", 128010 },
        { "</tool_call>", 128011 },
        { "</tool_response>", 128012 },
        { "<|eot_id|>", 128256 }
    };


void ChatML::tokenize() {
    auto newlineToken = ::llama_tokenize(ctx, "\n", false)[0];

    tokensList = { 128000 }; // bos_token
    for (auto message : messages) {
        tokensList.push_back(map["<|im_start|>"]);
        auto tokens = ::llama_tokenize(ctx, message.role, false);
        tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
        tokensList.push_back(newlineToken);
        tokens = ::llama_tokenize(ctx, message.content, false);
        tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
        tokensList.push_back(map["<|im_end|>"]);
        tokensList.push_back(newlineToken);
    }

    tokensList.push_back(map["<|im_start|>"]);
    auto tokens = ::llama_tokenize(ctx, "assistant", false);
    tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
    tokensList.push_back(newlineToken);
}

ChatML::ChatML(std::string modelPath, int seed, int nCtx, int nThreads, int nThreadsBatch, LLMType type, std::string systemMessage, int nGpuLayers, bool numa) 
: LLM(modelPath, seed, nCtx, nThreads, nThreadsBatch, type, systemMessage, nGpuLayers, numa) { }
