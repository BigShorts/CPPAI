#ifndef SRC_FORMATS_PHI3
#define SRC_FORMATS_PHI3
#include "../LLM.hpp"

class Phi3 : public LLM {
private:
    void tokenize() override;

public:
    Phi3(std::string modelPath, int seed, int nCtx, int nThreads, int nThreadsBatch, LLMType type, std::string systemMessage = "", int nGpuLayers = -1, bool numa = false);
};

#endif /* SRC_FORMATS_PHI3 */
