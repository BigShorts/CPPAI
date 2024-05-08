#ifndef SRC_FORMATS_CHATML
#define SRC_FORMATS_CHATML
#include "../LLM.hpp"

class ChatML : public LLM {
private:
    void tokenize() override;

public:
    ChatML(std::string modelPath, int seed, int nCtx, int nThreads, int nThreadsBatch, LLMType type, std::string systemMessage = "", int nGpuLayers = -1, bool numa = false);
};

#endif /* SRC_FORMATS_CHATML */
