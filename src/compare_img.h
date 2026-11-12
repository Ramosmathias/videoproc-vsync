#pragma once

#include <opencv2/core.hpp>

inline constexpr double invalidImageScore = -1.0;
inline constexpr double incompatibleImages = 2.0;

// Valeur maximale theorique d'une image (somme de tous les pixels a 255).
inline double MaxImgVal(const cv::Mat& img) { return double(img.total()) * img.channels() * 255.0; }

// Niveau moyen normalise de l'image (0..1).
// Concretement: calcule la moyenne de chaque canal (B,G,R...), additionne,
// puis normalise par (nb_canaux * 255). 0.0 = noir, 1.0 = blanc.
// Utilise pour reperer des pauses noires ou des variations brusques de luminosite.
inline double MeanLevel(const cv::Mat& img)
{
	if (img.empty()) return invalidImageScore;
	cv::Scalar m = cv::mean(img);
	int ch = img.channels();
	if (ch <= 0) return invalidImageScore;
	double sum = 0.0;
	for (int i = 0; i < ch; ++i) sum += m[i];
	return sum / (ch * 255.0);
}

// Version sans verification: a utiliser seulement si empty/size/type ont ete verifies en amont.
inline double CompareUnchecked(const cv::Mat& img1, const cv::Mat& img2, double invDenom)
{
	#ifdef use_norm_compare
	// Variante 1: somme des differences absolues en une seule operation OpenCV.
	return cv::norm(img1, img2, cv::NORM_L1) * invDenom;
	#else
	// Variante 2 : absdiff puis somme des canaux.
	thread_local cv::Mat difference;
	cv::absdiff(img1, img2, difference); // Calcule la difference absolue pixel par pixel.
	const cv::Scalar score = cv::sum(difference);
	double sumScore = 0.0; // Somme la difference pour chaque canal.
	for (int i = 0; i < img1.channels(); ++i) sumScore += score[i]; // Additionne toutes les composantes.
	return sumScore * invDenom;
	#endif
}

// Compare deux images et retourne un score normalise (0 identiques, 1 tres differentes).
inline double Compare(const cv::Mat& img1, const cv::Mat& img2)
{
	if (img1.empty() || img2.empty()) return invalidImageScore; // Verifie que les deux images existent.
	if (img1.size() != img2.size() || img1.type() != img2.type()) return incompatibleImages; // Refuse la comparaison si tailles/types differents.

	double denom = MaxImgVal(img1); // Calcule la somme maximale possible (normalisation).
	if (denom <= 0.0) return invalidImageScore;
	return CompareUnchecked(img1, img2, 1.0 / denom); // Score normalise dans [0,1] ou invalide.
}

// Variante optimisee: normalisation pre-calculee (invDenom = 1.0 / MaxImgVal(img1)).
inline double Compare(const cv::Mat& img1, const cv::Mat& img2, double invDenom)
{
	if (img1.empty() || img2.empty()) return invalidImageScore;
	if (img1.size() != img2.size() || img1.type() != img2.type()) return incompatibleImages;
	if (invDenom <= 0.0) return invalidImageScore;
	return CompareUnchecked(img1, img2, invDenom);
}
