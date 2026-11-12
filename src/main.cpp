#include "video_matching.h"

#include <charconv>
#include <chrono>
#include <iostream>
#include <string_view>

bool parseMilliseconds(std::string_view text, unsigned& milliseconds)
{
    const auto* end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, milliseconds);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

int main(int argc, char* argv[])
{
    unsigned referenceTime = 0;
    unsigned comparisonTime = 0;
    if (argc != 5 || !parseMilliseconds(argv[3], referenceTime)
        || !parseMilliseconds(argv[4], comparisonTime))
    {
        std::cerr << "Utilisation : videoproc <video1> <video2> <repere1_ms> <repere2_ms>\n";
        return 1;
    }

    try
    {
        VideoMatching alignment(argv[1], argv[2], referenceTime, comparisonTime);
        const auto start = std::chrono::steady_clock::now();
        if (!alignment.DynMatching() || !alignment.BackTracking())
        {
            std::cerr << "Impossible de calculer le chemin d'alignement.\n";
            return 2;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        if (!alignment.DumpScoresTxt("scores.txt"))
        {
            std::cerr << "Impossible d'exporter des segments dans scores.txt.\n";
            return 3;
        }
        std::cout << "Alignement temporel : " << elapsed.count() << " ms\n";
        std::cout << "Segments exportes dans scores.txt\n";
    }
    catch (const cv::Exception& error)
    {
        std::cerr << error.what() << '\n';
        return 4;
    }
}
