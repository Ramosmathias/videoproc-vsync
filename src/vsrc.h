#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>

#include "compare_img.h"

// Flux OpenCV et cache circulaire des images deja decodees.
class VideoStream
{
	struct FrameMetrics
	{
		double meanLevel = invalidImageScore;
		double absDelta = invalidImageScore;
	};

	cv::VideoCapture capture;
	std::vector<cv::Mat> imgBuf;
	cv::Mat rawFrame;
	cv::Mat frame;
	std::vector<FrameMetrics> frameData;
	unsigned nbImg = 0;
	unsigned currIdx = 0;
	unsigned nextBufIdx = 0;
	unsigned nbImgInBuf = 0;
	double tImg = 0.0;

	void ClearCache()
	{
		nextBufIdx = 0;
		nbImgInBuf = 0;
	}

public:
	static constexpr unsigned invalidIndex = std::numeric_limits<unsigned>::max();
	cv::Size originalSize;
	cv::Size outputSize;
	cv::Mat transform;

	// Convertit un temps en ms vers une chaine "hh:mm:ss.mmm".
	static std::string TimeMs2String(unsigned t_in_ms)
	{
		unsigned ms = t_in_ms % 1000;
		unsigned total_s = t_in_ms / 1000;
		unsigned s = total_s % 60;
		unsigned total_m = total_s / 60;
		unsigned m = total_m % 60;
		unsigned h = total_m / 60;
		char buf[32];
		std::snprintf(buf, sizeof(buf), "%02u:%02u:%02u.%03u", h, m, s, ms);
		return std::string(buf);
	}

	bool IsReadable() const { return nbImg != 0; }
	unsigned FrameCount() const { return nbImg; }

	bool SeekFrame(unsigned index)
	{
		if (index >= nbImg || !capture.set(cv::CAP_PROP_POS_FRAMES, index))
			return false;

		const double position = capture.get(cv::CAP_PROP_POS_FRAMES);
		if (!std::isfinite(position) || position != index)
			return false;

		currIdx = index;
		ClearCache();
		return true;
	}

	// Retourne le temps (ms) associe a l'index courant.
	unsigned CurrTimeMs() const { return Idx2TimeMs(currIdx); }
	// Convertit un temps (ms) en decalage d'index arrondi.
	signed TimeMs2RoundDIdx(signed t_in_ms) const
	{
		if (tImg <= 0.0) return 0;
		return (signed)std::lround(double(t_in_ms) / tImg);
	}
	// Convertit un temps (ms) en decalage d'index arrondi vers le haut.
	signed TimeMs2CeilDIdx(signed t_in_ms) const
	{
		if (tImg <= 0.0) return 0;
		return (signed)std::ceil(double(t_in_ms) / tImg);
	}
	// Convertit un temps (ms) en decalage d'index arrondi vers le bas.
	signed TimeMs2FloorDIdx(signed t_in_ms) const
	{
		if (tImg <= 0.0) return 0;
		return (signed)std::floor(double(t_in_ms) / tImg);
	}
	// Convertit un temps (ms) en index absolu (arrondi a l'image la plus proche).
	unsigned TimeMs2Idx(unsigned t_in_ms) const
	{
		if (tImg <= 0.0) return invalidIndex;
		return (unsigned)std::lround(double(t_in_ms) / tImg);
	}
	// Convertit un temps en index, puis sature dans [0, nbImg-1].
	unsigned TimeMs2SaturatedIdx(int t_in_ms) const
	{
		if (nbImg == 0) return 0;
		signed idx = TimeMs2RoundDIdx(t_in_ms);
		return SaturatedIdx(idx);
	}
	// Conversion securisee: si temps invalide, retourne l'ancien index.
	unsigned FrameAtTimeOr(int t_in_ms, unsigned oldIdx) const
	{
		if (t_in_ms < 0 || tImg <= 0.0) return oldIdx;
		return TimeMs2SaturatedIdx(t_in_ms);
	}
	// Sature un index dans l'intervalle [0, nbImg-1].
	unsigned SaturatedIdx(signed idx) const
	{
		if (nbImg == 0) return 0;
		if (idx < 0) return 0;
		unsigned u = (unsigned)idx;
		if (u >= nbImg) return nbImg - 1;
		return u;
	}
	// Retourne le delta t (ms) entre deux index d'images.
	unsigned Idx2TimeMs(unsigned idx1, unsigned idx0 = 0) const
	{
		if (idx1 < idx0 || tImg <= 0.0) return 0;
		double dt = double(idx1 - idx0) * tImg;
		return (unsigned)std::lround(dt);
	}
	// Convertit un index (ou delta d'index) vers "hh:mm:ss.mmm".
	std::string Idx2String(unsigned idx1, unsigned idx0 = 0) const
	{
		return TimeMs2String(Idx2TimeMs(idx1, idx0));
	}

	explicit VideoStream(const char* filename)
		: capture(filename, cv::CAP_FFMPEG)
	{
		if (!capture.isOpened())
		{
			std::cerr << "Impossible d'ouvrir le flux " << filename << '\n';
			return;
		}

		const double frameRate = capture.get(cv::CAP_PROP_FPS);
		const double frameCount = capture.get(cv::CAP_PROP_FRAME_COUNT);
		originalSize.width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
		originalSize.height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
		if (!std::isfinite(frameRate) || frameRate <= 0.0 ||
			!std::isfinite(frameCount) || frameCount < 1.0 || frameCount >= invalidIndex ||
			originalSize.empty())
		{
			std::cerr << "Metadonnees video inutilisables pour " << filename << '\n';
			return;
		}

		outputSize = originalSize;
		transform = cv::Mat::eye(2, 3, CV_64F);
		tImg = 1000.0 / frameRate;
		nbImg = static_cast<unsigned>(frameCount);
		frameData.resize(nbImg);
		std::cout << filename << " : " << nbImg << " images de " << originalSize
				  << " a " << frameRate << " images/s\n";
	}

	void SetCacheCapacity(unsigned capacity)
	{
		imgBuf.resize(capacity);
		ClearCache();
	}

	// ATTENTION: ne jamais confondre l'index video (index absolu d'image)
	// avec l'index de tampon circulaire (emplacement de stockage dans le cache) !
	// Acces a une image par index si elle est encore dans le tampon (sinon null).
	cv::Mat* operator[](unsigned idx)
	{
		if (imgBuf.empty())
		{
			if (currIdx == 0) return 0;
			return (idx + 1 == currIdx) ? &frame : 0;
		}
		if (nbImgInBuf == 0) return 0;
		unsigned firstIdx = currIdx - nbImgInBuf;
		if (idx < firstIdx || idx >= currIdx) return 0;
		unsigned offset = idx - firstIdx;
		// firstBufIdx = slot de l'image la plus ancienne dans le ring.
		// Cette formule reste correcte que le buffer soit plein ou non.
		const auto bufSz = imgBuf.size();
		const auto firstBufIdx = (nextBufIdx + bufSz - nbImgInBuf) % bufSz;
		const auto bufIdx = (firstBufIdx + offset) % bufSz;
		return &imgBuf[bufIdx];
	}
private:
	// Lit une nouvelle image, redimensionne si besoin et l'ecrit dans dst [appele seulement par NextImg].
	// Lit l'image suivante, applique la transformation affine si necessaire et ecrit dans dst.
	bool ReadResize(cv::Mat& dst)
	{
		if (!capture.read(rawFrame)) return false;
		if (originalSize != outputSize)
			cv::warpAffine(rawFrame, dst, transform, outputSize, cv::WARP_INVERSE_MAP);
		else std::swap(dst, rawFrame);
		return true;
	}
public:
	// Lit la prochaine image, met a jour le tampon circulaire et calcule les statistiques.
	cv::Mat* NextImg()
	{
		if (currIdx >= nbImg) return 0;
		cv::Mat* last = 0;
		unsigned idx = currIdx;
		if (imgBuf.empty())
		{
			if (!ReadResize(frame)) return 0;
			last = &frame;
		}
		else
		{
			cv::Mat& dst = imgBuf[nextBufIdx]; // Slot du tampon ou ecrire la nouvelle image.
			if (!ReadResize(dst)) return 0; // Lit et ecrit l'image (avec transformation si besoin).
			nextBufIdx = (nextBufIdx + 1) % imgBuf.size();
			if (nbImgInBuf < imgBuf.size()) ++nbImgInBuf; // Incremente tant que le tampon n'est pas plein.
			last = &dst;
		}
		++currIdx; // On a lu une image, donc l'index courant avance.
		if (idx < frameData.size()) // Verifie qu'on peut stocker les stats pour cet index.
		{
			double m = MeanLevel(*last); // Calcule la luminosite moyenne de l'image lue.
			frameData[idx].meanLevel = m; // Stocke la moyenne.
			if (idx > 0 && frameData[idx - 1].meanLevel != invalidImageScore && m != invalidImageScore) // Si l'image precedente est valide.
				frameData[idx].absDelta = std::fabs(m - frameData[idx - 1].meanLevel); // Stocke la variation de luminosite.
			else
				frameData[idx].absDelta = invalidImageScore; // Sinon, valeur invalide.
		}
		return last;

	}

	// Avance de n images en appelant NextImg n fois.
	bool NextImgs(unsigned n)
	{
		for (unsigned i = 0; i < n; ++i)
			if (!NextImg()) return false;
		return true;
	}

	// Ecrit un fichier texte avec mean_level et abs_delta pour une plage d'images.
	bool WriteFrameMetrics(const char* fn_out, unsigned end_idx = 0, unsigned start_idx = 0) const
	{
		if (!fn_out || nbImg == 0) return false;
		if (start_idx >= nbImg) return false;
		unsigned last_idx = (end_idx == 0 || end_idx >= nbImg) ? (nbImg - 1) : end_idx;
		if (last_idx < start_idx) return false;
		std::ofstream os(fn_out);
		if (!os) return false;
		os << "idx\ttemps_ms\tniveau_moyen\tdelta_abs\n";
		for (unsigned idx = start_idx; idx <= last_idx; ++idx)
		{
			const FrameMetrics& fd = frameData[idx];
			os << idx << "\t" << Idx2TimeMs(idx) << "\t" << fd.meanLevel << "\t" << fd.absDelta << "\n";
		}
		return true;
	}
};
