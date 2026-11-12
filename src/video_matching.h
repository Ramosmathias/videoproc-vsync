#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/optim.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <utility>

#include "vsrc.h"

// Recalage spatial initial et alignement temporel a bande adaptative.
class VideoMatching
{
    enum class Step : unsigned char { None, Horizontal, Vertical, Diagonal };

    struct Cell
    {
        double score = 1.0;
        double cumsum = invalidImageScore;
        Step link = Step::None;
        bool isOnTrack = false;
    };

    static constexpr double diagonalTurnPenalty = 0.04;
    static constexpr double rightAngleTurnPenalty = 0.08;
    static constexpr double horizontalPenalty = 0.015;
    static constexpr double verticalPenalty = 0.015;
    static constexpr double unreachableCost = 1e10;
    static constexpr int initialRadiusMs = 500;
    static constexpr int fallbackAnchorMs = 2000;

    std::vector<std::vector<Cell>> Matrice;
    VideoStream vSrc1;
    VideoStream vSrc2;
    VideoStream* pSrc1 = nullptr;
    VideoStream* pSrc2 = nullptr;
    bool isOK = false;

    // Couple d'index de synchro utilise comme ancrage initial pour le matching dynamique.
    unsigned syncAnchorHRIdx;
    unsigned syncAnchorLRIdx;
    bool hasSyncAnchor;

    // Stockage sparse de la matrice:
    // - Matrice[r] contient uniquement la bande calculee pour la ligne r
    // - matStartLR[r] donne l'indice LR absolu de la 1ere case de Matrice[r]
    // - matNbLRCols memorise le nombre total de colonnes LR du probleme.
    std::vector<unsigned> matStartLR;
    unsigned matNbLRCols;

    // Verifie si la cellule absolue (r,c) existe dans la matrice sparse calculee.
    // Retourne false si la ligne n'existe pas ou si la colonne est hors bande.
    bool HasCell(unsigned r, unsigned c) const
    {
        if (r >= (unsigned)Matrice.size() || r >= (unsigned)matStartLR.size()) return false;
        const unsigned c0 = matStartLR[r];
        if (c < c0) return false;
        const unsigned lc = c - c0;
        return lc < (unsigned)Matrice[r].size();
    }

    // Acces en lecture seule a une cellule absolue (r,c) dans la matrice sparse.
    // Retourne nullptr si la cellule n'a pas ete calculee.
    const Cell* GetCell(unsigned r, unsigned c) const
    {
        if (!HasCell(r, c)) return 0;
        return &Matrice[r][c - matStartLR[r]];
    }

    // Acces modifiable a une cellule absolue (r,c) dans la matrice sparse.
    // Retourne nullptr si la cellule n'a pas ete calculee.
    Cell* GetCell(unsigned r, unsigned c)
    {
        if (!HasCell(r, c)) return 0;
        return &Matrice[r][c - matStartLR[r]];
    }

    // Borne une valeur d'indice dans [0, maxIdx].
    static unsigned ClampToRow(signed long long v, unsigned maxIdx)
    {
        if (v < 0) return 0u;
        if ((unsigned long long)v > (unsigned long long)maxIdx) return maxIdx;
        return (unsigned)v;
    }

    // Repositionne le flux LR a un indice absolu et reinitialise l'etat du decodeur sequentiel.
    bool ResetLRStreamAt(unsigned idx, unsigned& lrNextExpected, bool& hasLRLoaded)
    {
        if (!pSrc2->SeekFrame(idx)) return false;
        lrNextExpected = idx;
        hasLRLoaded = false;
        return true;
    }

    // Decode une image LR supplementaire en lecture sequentielle.
    bool DecodeNextLR(unsigned& lrLoadedUntil, unsigned& lrNextExpected, bool& hasLRLoaded)
    {
        cv::Mat* p = pSrc2->NextImg();
        if (!p) return false;
        lrLoadedUntil = lrNextExpected;
        ++lrNextExpected;
        hasLRLoaded = true;
        return true;
    }

    // Garantit que toutes les images LR jusqu'a idx sont decodees.
    bool EnsureLRDecodedTo(unsigned idx, unsigned& lrLoadedUntil, unsigned& lrNextExpected, bool& hasLRLoaded)
    {
        while (!hasLRLoaded || lrLoadedUntil < idx)
            if (!DecodeNextLR(lrLoadedUntil, lrNextExpected, hasLRLoaded)) return false;
        return true;
    }

    // En cas d'echec de decodage LR, reduit la borne droite effective des colonnes LR.
    // Retourne true si la borne reduite permet de retenter utilement la ligne courante.
    bool ClampEffectiveTailOnDecodeFail(
        unsigned reqEnd,
        unsigned rowStart,
        unsigned nbLR,
        unsigned& effectiveMaxLR,
        unsigned& centerLRIdx,
        bool hasLRLoaded,
        unsigned lrLoadedUntil,
        unsigned lrNextExpected,
        bool& hasLoggedEffectiveTail)
    {
        unsigned newMax = effectiveMaxLR;
        if (hasLRLoaded) newMax = lrLoadedUntil;
        else if (lrNextExpected > 0u) newMax = lrNextExpected - 1u;
        else if (rowStart > 0u) newMax = rowStart - 1u;
        else newMax = 0u;

        if (newMax >= effectiveMaxLR) return false;

        effectiveMaxLR = newMax;
        matNbLRCols = effectiveMaxLR + 1u;
        if (centerLRIdx > effectiveMaxLR) centerLRIdx = effectiveMaxLR;

        #ifdef verbose
        if (!hasLoggedEffectiveTail)
        {
            std::cout << "[Dyn] info: borne LR effective reduite a " << effectiveMaxLR
                      << " (nbLR annonce=" << nbLR << ")\n";
            hasLoggedEffectiveTail = true;
        }
        #endif

        return (effectiveMaxLR >= rowStart && effectiveMaxLR < reqEnd);
    }

    // Evalue un candidat DP (N, O ou NO) et met a jour le meilleur cout/lien si necessaire.
    static void TryCandidate(
        double localScore,
        double prevCost,
        Step prevLink,
        Step newLink,
        double validCumSumTh,
        double& bestCandidate,
        Step& bestLink,
        bool& hasCandidate)
    {
        if (prevCost >= validCumSumTh) return;

        double extra = 0.0;
        if (newLink == Step::Horizontal) extra += horizontalPenalty;
        else if (newLink == Step::Vertical) extra += verticalPenalty;

        if (prevLink != Step::None && prevLink != newLink)
        {
            if ((prevLink == Step::Horizontal && newLink == Step::Vertical) ||
                (prevLink == Step::Vertical && newLink == Step::Horizontal))
                extra += rightAngleTurnPenalty;
            else
                extra += diagonalTurnPenalty;
        }

        const double cand = localScore + prevCost + extra;
        if (!hasCandidate || cand < bestCandidate)
        {
            bestCandidate = cand;
            bestLink = newLink;
            hasCandidate = true;
        }
    }

public:
    // Constructeur principal:
    // - ouvre les 2 flux,
    // - permute eventuellement pour placer HR en source 1 et LR en source 2,
    // - convertit les timestamps de synchro en indices,
    // - lance l'optimisation de recalage spatial initiale.
    VideoMatching(const char* vname1, const char* vname2, unsigned tsync1, unsigned tsync2) :
        vSrc1(vname1), vSrc2(vname2),
        syncAnchorHRIdx(0), syncAnchorLRIdx(0), hasSyncAnchor(false),
        matNbLRCols(0)
    {
        if (!vSrc1.IsReadable() || !vSrc2.IsReadable()) return;
        pSrc1 = &vSrc1; pSrc2 = &vSrc2;
        if (vSrc1.originalSize.width < vSrc2.originalSize.width)
        {
            std::swap(pSrc1, pSrc2);
            std::swap(tsync1, tsync2);
        }

        pSrc1->SetCacheCapacity(1);    // HR: une image suffit pour la ligne courante
        pSrc2->SetCacheCapacity(256);  // LR: petit cache glissant

        // Convertit les temps de synchro en index d'images, puis lance l'optimisation spatiale.
        const unsigned idxSyncHR = pSrc1->TimeMs2Idx(tsync1);
        const unsigned idxSyncLR = pSrc2->TimeMs2Idx(tsync2);
        if (!AlignAnchorFrames(idxSyncHR, idxSyncLR)) return;
        isOK = true;
    }

    class SpatialCost final : public cv::MinProblemSolver::Function
    {
        const cv::Mat& source;
        const cv::Mat& reference;
        mutable cv::Mat aligned;

    public:
        SpatialCost(const cv::Mat& source, const cv::Mat& reference)
            : source(source), reference(reference) {}

        int getDims() const override { return 4; }

        double calc(const double* parameters) const override
        {
            const cv::Matx23d transform(parameters[0], 0.0, parameters[1],
                                        0.0, parameters[2], parameters[3]);
            cv::warpAffine(source, aligned, transform, reference.size(), cv::WARP_INVERSE_MAP);
            return Compare(aligned, reference);
        }
    };

    bool AlignAnchorFrames(unsigned idx1, unsigned idx2)
    {
        std::cout << "Synchro spatiale aux points " << idx1 << "<->" << idx2 << "\n";
        if (!pSrc1->SeekFrame(idx1)) return false;
        if (!pSrc2->SeekFrame(idx2)) return false;

        cv::Mat* m1p = pSrc1->NextImg();
        cv::Mat* m2p = pSrc2->NextImg();
        if (!m1p || !m2p) return false;
        cv::Mat& m1 = *m1p;
        cv::Mat& m2 = *m2p;

        // Parametres initiaux: (sx, tx, sy, ty) pour warpAffine.
        double par[] = {
            double(pSrc1->outputSize.width) / double(pSrc2->outputSize.width), 0,
            double(pSrc1->outputSize.height) / double(pSrc2->outputSize.height), 0
        };
        double par_init[] = { par[0], par[1], par[2], par[3] };
        double step[] = { 0.01, 1.0, 0.01, 1.0 };

        const auto fm = cv::makePtr<SpatialCost>(m1, m2);
        double score_init = fm->calc(par_init);

        cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create(fm, cv::Mat(4, 1, CV_64FC1, (void*)step));
        cv::TermCriteria termCrit(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 5000, 1e-6);
        solver->setTermCriteria(termCrit);

        cv::Mat optim(4, 1, CV_64FC1, (void*)par);
        double minValue = solver->minimize(optim);
        double score_final = fm->calc(par);

        std::cout << "Synchro spatiale: score initial=" << score_init << " final=" << score_final << " min=" << minValue << "\n";
        std::cout << "Parametres : sx=" << par[0] << " tx=" << par[1] << " sy=" << par[2] << " ty=" << par[3] << "\n";

        pSrc1->outputSize = pSrc2->outputSize;
        pSrc1->transform = (cv::Mat_<double>(2, 3) << par[0], 0.0, par[1], 0.0, par[2], par[3]);
        syncAnchorHRIdx = idx1;
        syncAnchorLRIdx = idx2;
        hasSyncAnchor = true;
        return pSrc1->SeekFrame(0) && pSrc2->SeekFrame(0);
    }

    // Remplit la matrice DP (ou une sous-matrice continue):
    // axe Y = indices HR, axe X = indices LR.
    // cumsum(i,j) = score(i,j) + min(N, NO, O), voisins hors matrice => 0.
    // Si appelee sans argument: start=0, end=last => toute la matrice.
    bool DynMatching(unsigned startHRIdx = 0, unsigned endHRIdx = VideoStream::invalidIndex)
    {
        #ifdef verbose
        std::cout << "[Dyn] entree startHRIdx=" << startHRIdx << ", endHRIdx=" << endHRIdx << "\n";
        #endif
        if (!isOK)
        {
            #ifdef verbose
            std::cout << "[Dyn] echec: objet non initialise (isOK=false)\n";
            #endif
            return false;
        }

        const unsigned nbHR = pSrc1->FrameCount();
        const unsigned nbLR = pSrc2->FrameCount();
        if (nbHR == 0 || nbLR == 0)
        {
            #ifdef verbose
            std::cout << "[Dyn] echec: flux vide(s), nbHR=" << nbHR << ", nbLR=" << nbLR << "\n";
            #endif
            return false;
        }

        const unsigned start = startHRIdx;
        if (start >= nbHR)
        {
            #ifdef verbose
            std::cout << "[Dyn] echec: indice de debut hors plage, debut=" << start << ", nbHR=" << nbHR << "\n";
            #endif
            return false;
        }

        unsigned end = endHRIdx;
        if (end == VideoStream::invalidIndex || end >= nbHR) end = nbHR - 1;
        if (end < start)
        {
            #ifdef verbose
            std::cout << "[Dyn] echec: fin < debut, debut=" << start << ", end=" << end << "\n";
            #endif
            return false;
        }

        // Nombre de lignes HR a calculer: start, start+1, ... <= end.
        const unsigned rowCount = (end - start + 1u);
        #ifdef verbose
        std::cout << "[Dyn] configuration nbHR=" << nbHR << ", nbLR=" << nbLR
                      << ", start=" << start << ", end=" << end
                      << ", nbLignes=" << rowCount << "\n";
        #endif
        Matrice.clear();
        Matrice.resize(rowCount);
        matStartLR.assign(rowCount, 0u);
        matNbLRCols = nbLR;

        // Pointeurs vers les images courantes lues dans les flux HR et LR.
        cv::Mat* pHR = 0;
        cv::Mat* pLR = 0;
        unsigned iHR = start;
        unsigned j = 0;
        // Alias pointeur vers la ligne DP courante (Matrice[r].data()) pour acces rapide aux cases.
        Cell* rowData = 0;
        const unsigned alwaysLogStepRows = 5000u;
        // Seuil utilitaire: au-dela, un cumsum est considere comme non exploitable (INF/invalides).
        const double validCumSumTh = unreachableCost * 0.5;
        // Inverse du denominateur de normalisation des scores de comparaison (pre-calcule une fois).
        double invDenom = 0.0;
        // Rayon de securite exprime en indices LR, derive de initialRadiusMs (en ms).
        const signed safeRayDIdx = pSrc2->TimeMs2RoundDIdx(initialRadiusMs);
        // Valeur absolue du rayon de securite en indices.
        const unsigned safeRayIdx = (safeRayDIdx < 0) ? (unsigned)(-safeRayDIdx) : (unsigned)safeRayDIdx;
        // Rayon minimal de bande utilise par defaut (au moins 24).
        const unsigned minBandRay = (safeRayIdx < 24u) ? 24u : safeRayIdx; // ~25 colonnes min.
        // Rayon maximal theorique (borne par la largeur LR disponible).
        const unsigned maxBandRay = (nbLR > 0u) ? (nbLR - 1u) : 0u;
        // Rayon maximal applique en pratique (cap hard a 255 pour contenir le cout).
        const unsigned hardMaxRay = (maxBandRay < 255u) ? maxBandRay : 255u;
        // on agrandit le rayon quand le minimum touche le bord de bande.
        const unsigned growStep = 5u;
        // on reduit le rayon quand la trajectoire reste stable.
        const unsigned shrinkStep = 1u;
        // Marge de bord a partir de laquelle on considere que la bande est trop serree.
        const unsigned edgeGrowMargin = 2u;
        // Marge de centrage a respecter pour qualifier une ligne de "stable".
        const unsigned centerStableMargin = 8u;
        // Nombre de lignes stables consecutives avant d'autoriser une reduction du rayon.
        const unsigned stableRowsToShrink = 4u;
        // Rayon courant effectif de la bande pour la ligne en cours.
        unsigned currBandRay = minBandRay;
        // Compteur de lignes consecutives ou le minimum reste bien centre.
        unsigned stableCenteredRows = 0u;
        // Colonne LR centrale utilisee pour positionner la bande de la ligne courante.
        unsigned centerLRIdx = 0u;
        // Centre predit par le chemin (bestCol de la ligne precedente).
        unsigned pathPredCenter = 0u;
        // Debut de bande (colonne LR absolue) pour la ligne courante.
        unsigned jStart = 0u;
        // Fin de bande (colonne LR absolue) pour la ligne courante.
        unsigned jEnd = 0u;
        // Indique si un minimum valide a deja ete trouve sur la ligne precedente.
        bool hasPrevBest = false;
        // Colonne LR du minimum de la ligne precedente.
        unsigned prevBestCol = 0u;
        // Indique si le decodeur LR a deja charge au moins une image dans l'etat courant.
        bool hasLRLoaded = false;

		// Dernier indice LR effectivement decode et disponible.
        unsigned lrLoadedUntil = 0u;
        // Prochain indice LR attendu en lecture sequentielle.
        unsigned lrNextExpected = 0u;
        // Derniere colonne LR encore consideree utilisable (peut diminuer si fin reelle detectee).
        unsigned effectiveMaxLR = nbLR - 1u;

		// Evite de reafficher plusieurs fois le meme message de reduction de borne LR.
        bool hasLoggedEffectiveTail = false;

		// memoire Cache des pointeurs d'images LR de la bande courante.
        std::vector<const cv::Mat*> bandFrames;
        // memoire Cache des scores locaux Compare(HR, LR) pour la bande courante.
        std::vector<double> bandScores;

		// Colonne LR du meilleur cumsum pour chaque ligne HR calculee.
        std::vector<unsigned> bestColRow(rowCount, VideoStream::invalidIndex);
        // Valeur de meilleur cumsum pour chaque ligne HR calculee.
        std::vector<double> bestCostRow(rowCount, unreachableCost);

        // Les operations de gestion LR (bornage, reset/decode, adaptation de borne) sont
        // factorisees dans des methodes privees de la classe pour eviter les lambdas locales.

        // Verification unique avant la grosse boucle:
        // - existence d'une image HR/LR
        // - compatibilite size/type
        // - denom valide pour la normalisation
        if (!pSrc1->SeekFrame(start))
        {
            #ifdef verbose
            std::cout << "[Dyn] echec: precontrole SeekFrame(HR," << start << ") en echec\n";
            #endif
            return false;
        }
        pHR = pSrc1->NextImg();
        if (!pHR)
        {
            #ifdef verbose
            std::cout << "[Dyn] echec: precontrole NextImg(HR) en echec a l'indice " << start << "\n";
            #endif
            return false;
        }
        if (!pSrc2->SeekFrame(0))
        {
            #ifdef verbose
            std::cout << "[Dyn] echec: precontrole SeekFrame(LR,0) en echec\n";
            #endif
            return false;
        }
        pLR = pSrc2->NextImg();
        if (!pLR)
        {
            #ifdef verbose
            std::cout << "[Dyn] echec: precontrole NextImg(LR) en echec a l'indice 0\n";
            #endif
            return false;
        }
        if (pHR->empty() || pLR->empty())
        {
            #ifdef verbose
            std::cout << "[Dyn] echec: precontrole image(s) vide(s)\n";
            #endif
            return false;
        }
        if (pHR->size() != pLR->size() || pHR->type() != pLR->type())
        {
            #ifdef verbose
            std::cout << "[Dyn] echec: precontrole incompatibilite taille/type entre HR et LR\n";
            #endif
            return false;
        }
        {
            const double denom = MaxImgVal(*pHR);
            if (denom <= 0.0)
            {
                #ifdef verbose
                std::cout << "[Dyn] echec: denominateur de normalisation invalide au precontrole\n";
                #endif
                return false;
            }
            invDenom = 1.0 / denom;
        }

        // Positionne les flux au debut de la plage, puis avance ensuite de facon monotone.
        if (!pSrc1->SeekFrame(start))
        {
            #ifdef verbose
            std::cout << "[Dyn] echec: initialisation HR SeekFrame(" << start << ") en echec\n";
            #endif
            return false;
        }
        if (!ResetLRStreamAt(0u, lrNextExpected, hasLRLoaded))
        {
            #ifdef verbose
            std::cout << "[Dyn] echec: initialisation LR SeekFrame(0) en echec\n";
            #endif
            return false;
        }

        // Initialise le centre LR de la bande:
        // - si une ancre de synchro existe, on projette le centre depuis ce couple (HR,LR),
        // - sinon on utilise une estimation temporelle par defaut puis on borne l'indice.
        if (hasSyncAnchor)
        {
            const signed long long delta = (signed long long)start - (signed long long)syncAnchorHRIdx;
            centerLRIdx = ClampToRow((signed long long)syncAnchorLRIdx + delta, nbLR - 1u);
        }
        else
        {
            centerLRIdx = pSrc2->FrameAtTimeOr(fallbackAnchorMs, 0);
            if (centerLRIdx >= nbLR) centerLRIdx = nbLR - 1u;
            if (centerLRIdx > effectiveMaxLR) centerLRIdx = effectiveMaxLR;
        }

        // Boucle principale DP:
        // pour chaque ligne HR, positionne la bande LR, calcule les scores locaux,
        // puis propage les cumsums et adapte la bande pour la ligne suivante.
        for (unsigned r = 0; r < rowCount; ++r, ++iHR)
        {
            // Etape 1: lire l'image HR courante (avance monotone).
            // HR monotone: une seule lecture sequentielle par ligne.
            pHR = pSrc1->NextImg();
            if (!pHR)
            {
                #ifdef verbose
                std::cout << "[Dyn] echec: NextImg(HR) en echec a l'indice " << iHR << "\n";
                #endif
                return false;
            }

            // Etape 2: definir le centre LR de travail pour cette ligne.
            // Centre pilote uniquement par le chemin:
            // ligne 0 = centre initial deja calcule via ancrage/fallback avant la boucle,
            // puis (r>0) bestCol de la ligne precedente.
            if (r != 0u)
            {
                pathPredCenter = hasPrevBest ? prevBestCol : centerLRIdx;
                centerLRIdx = pathPredCenter;
            }
            if (centerLRIdx >= nbLR) centerLRIdx = nbLR - 1u;

            // Nombre de cases effectivement calculees (cumsum valide) dans la bande courante.
            unsigned computedInBand = 0u;
            // Indique si un minimum de ligne exploitable a ete trouve.
            bool rowBestFound = false;
            // Colonne LR (absolue) du meilleur minimum trouve sur la ligne.
            unsigned rowBestCol = centerLRIdx;
            // Cout cumule minimal de la ligne (initialise a une grande valeur).
            double rowBestCost = unreachableCost;
            // Active le repli "ligne complete" si la bande echoue.
            bool fallbackFullRow = false;

            // Etape 3: calculer la ligne r (avec eventuel repli "ligne complete").
            for (;;)
            {
                // Etape 3.1: construire la bande LR [jStart, jEnd] et initialiser les cases de la ligne.
                if (fallbackFullRow)
                {
                    jStart = 0u;
                    jEnd = effectiveMaxLR;
                }
                else
                {
                    // Bande centree sur centerLRIdx:
                    // - debut borne a 0 pour eviter les indices negatifs,
                    // - fin bornee a effectiveMaxLR (et protection overflow unsigned).
                    jStart = (centerLRIdx > currBandRay) ? (centerLRIdx - currBandRay) : 0u;
                    jEnd = centerLRIdx + currBandRay;
                    if (jEnd < centerLRIdx || jEnd > effectiveMaxLR) jEnd = effectiveMaxLR;
                }

                if (jStart > jEnd) jStart = jEnd;
                const unsigned bandCols = (jEnd - jStart + 1u);
                Matrice[r].resize(bandCols);
                matStartLR[r] = jStart;
                bandFrames.resize(bandCols);
                bandScores.resize(bandCols);
                rowData = Matrice[r].data();
                for (unsigned k = 0; k < bandCols; ++k)
                {
                    rowData[k].score = incompatibleImages;
                    rowData[k].cumsum = unreachableCost;
                    rowData[k].link = Step::None;
                    rowData[k].isOnTrack = false;
                }

                #ifdef verbose
                std::cout << "[Dyn] ligne " << r << "/" << (rowCount - 1)
                              << " (HR idx=" << iHR << ", colonnes=" << matNbLRCols
                              << ", centre=" << centerLRIdx << ", rayon=" << currBandRay
                              << ", bande=[" << jStart << "," << jEnd << "]"
                              << (fallbackFullRow ? ", repli=ligne-complete" : "") << ")\n";
                #endif

                // Etape 3.2: preparer l'acces aux images LR (buffer circulaire prioritaire, sinon streaming).
                // LR: tente d'utiliser le buffer circulaire (pas de redecodage massif).
                bool useStreamingRow = fallbackFullRow;
                bool bandFramesReady = false;
                if (!useStreamingRow)
                {
                    // Decide s'il faut repositionner LR sur jStart:
                    // - reset si la prochaine lecture attendue ne tombe pas sur jStart,
                    // - sinon on continue en lecture sequentielle sans SeekFrame.
                    bool needReset = false;
                    if (!hasLRLoaded) needReset = (lrNextExpected != jStart);
                    else needReset = (jStart > (lrLoadedUntil + 1u));

                    if (needReset && !ResetLRStreamAt(jStart, lrNextExpected, hasLRLoaded))
                    {
                        #ifdef verbose
                        std::cout << "[Dyn] echec: reinitialisation LR SeekFrame(" << jStart << ") en echec a la ligne " << r << "\n";
                        #endif
                        return false;
                    }
                    // Si on n'arrive pas a decoder jusqu'a jEnd, on reduit la borne LR effective
                    // a la fin reellement lisible puis on retente la meme ligne.
                    if (!EnsureLRDecodedTo(jEnd, lrLoadedUntil, lrNextExpected, hasLRLoaded))
                    {
                        if (ClampEffectiveTailOnDecodeFail(
                                jEnd,
                                jStart,
                                nbLR,
                                effectiveMaxLR,
                                centerLRIdx,
                                hasLRLoaded,
                                lrLoadedUntil,
                                lrNextExpected,
                                hasLoggedEffectiveTail))
                            continue;
                        #ifdef verbose
                        std::cout << "[Dyn] avertissement: decodage LR jusqu'a " << jEnd << " en echec, repli lecture sequentielle ligne " << r << "\n";
                        #endif
                        useStreamingRow = true;
                    }
                    else
                    {
                        // Etape A: verifier si toute la bande LR est deja presente dans le buffer circulaire.
                        bool allInBuf = true;
                        for (unsigned lk = 0u; lk < bandCols; ++lk)
                        {
                            const cv::Mat* pBand = (*pSrc2)[jStart + lk];
                            bandFrames[lk] = pBand;
                            if (!pBand) { allInBuf = false; break; }
                        }
                        if (!allInBuf)
                        {
                            // Etape B: si la bande n'est pas complete en buffer, forcer une remise en place
                            // puis decoder jusqu'a jEnd, et retester le buffer une seconde fois.
                            if (ResetLRStreamAt(jStart, lrNextExpected, hasLRLoaded) &&
                                EnsureLRDecodedTo(jEnd, lrLoadedUntil, lrNextExpected, hasLRLoaded))
                            {
                                allInBuf = true;
                                for (unsigned lk = 0u; lk < bandCols; ++lk)
                                {
                                    const cv::Mat* pBand = (*pSrc2)[jStart + lk];
                                    bandFrames[lk] = pBand;
                                    if (!pBand) { allInBuf = false; break; }
                                }
                            }
                            // Etape C: en cas d'echec de decodage, reduire la borne LR effective
                            // et retenter la ligne courante avec la nouvelle borne.
                            else if (ClampEffectiveTailOnDecodeFail(
                                         jEnd,
                                         jStart,
                                         nbLR,
                                         effectiveMaxLR,
                                         centerLRIdx,
                                         hasLRLoaded,
                                         lrLoadedUntil,
                                         lrNextExpected,
                                         hasLoggedEffectiveTail))
                            {
                                continue;
                            }
                            // Etape D: si la bande n'est toujours pas complete en buffer,
                            // basculer en mode lecture sequentielle (streaming) pour cette ligne.
                            if (!allInBuf) useStreamingRow = true;
                        }
                        else
                        {
                            // Etape E: bande complete en buffer, prete pour le calcul des scores locaux.
                            bandFramesReady = true;
                        }
                    }
                }
                if (useStreamingRow)
                {
                    // Repli: repositionnement direct sur le debut de bande.
                    if (!pSrc2->SeekFrame(jStart))
                    {
                        #ifdef verbose
                        std::cout << "[Dyn] echec: SeekFrame(LR," << jStart << ") en echec a la ligne " << r << "\n";
                        #endif
                        return false;
                    }
                }

                computedInBand = 0u;
                rowBestFound = false;
                rowBestCost = unreachableCost;
                rowBestCol = centerLRIdx;

                // Etape 3.3: recuperer les informations de la ligne precedente (voisins Nord et Nord-Ouest).
                const Cell* prevRowData = 0;
                unsigned prevRowStart = 0u;
                unsigned prevRowEnd = 0u;
                if (r > 0u && !Matrice[r - 1u].empty())
                {
                    prevRowData = Matrice[r - 1u].data();
                    prevRowStart = matStartLR[r - 1u];
                    prevRowEnd = prevRowStart + (unsigned)Matrice[r - 1u].size() - 1u;
                }

                // Phase A: charge les images LR de la bande.
                unsigned loadedCount = 0u;
                if (useStreamingRow)
                {
                    for (; loadedCount < bandCols; ++loadedCount)
                    {
                        pLR = pSrc2->NextImg();
                        if (!pLR) break;
                        bandFrames[loadedCount] = pLR;
                    }
                }
                else if (bandFramesReady)
                {
                    loadedCount = bandCols;
                }
                else
                {
                    for (; loadedCount < bandCols; ++loadedCount)
                    {
                        const cv::Mat* pBand = (*pSrc2)[jStart + loadedCount];
                        if (!pBand) break;
                        bandFrames[loadedCount] = pBand;
                    }
                }
                j = jStart + loadedCount;

                // Phase B: calcule les scores locaux (parallele si possible).
                if (loadedCount > 0u)
                {
                    #if defined(_OPENMP) && defined(use_norm_compare)
                    #pragma omp parallel for
                    for (int lk = 0; lk < (int)loadedCount; ++lk)
                    {
                        const cv::Mat* pLRRow = bandFrames[(unsigned)lk];
                        bandScores[(unsigned)lk] = pLRRow ? CompareUnchecked(*pHR, *pLRRow, invDenom) : invalidImageScore;
                    }
                    #else
                    for (unsigned lk = 0u; lk < loadedCount; ++lk)
                    {
                        const cv::Mat* pLRRow = bandFrames[lk];
                        bandScores[lk] = pLRRow ? CompareUnchecked(*pHR, *pLRRow, invDenom) : invalidImageScore;
                    }
                    #endif
                }

                // Phase C: propagation DP (sequentielle, depend de Ouest).
                for (unsigned localJ = 0u; localJ < loadedCount; ++localJ)
                {
                    // C.1: position LR absolue et case courante de la ligne DP.
                    const unsigned jAbs = jStart + localJ;
                    Cell& cd = rowData[localJ];

                    // C.2: injecter le score local pre-calcule pour cette case (HR, LR).
                    cd.score = bandScores[localJ];

                    // C.3: rejeter les comparaisons invalides/non appariables.
                    if (cd.score == invalidImageScore || cd.score == incompatibleImages)
                    {
                        cd.cumsum = unreachableCost;
                        cd.link = Step::None;
                        continue;
                    }

                    // C.4: initialiser la recherche du meilleur predecesseur.
                    double bestCandidate = unreachableCost;
                    Step bestLink = Step::None;
                    bool hasCandidate = false;

                    // C.5: tester le voisin Nord-Ouest (lien diagonal DLink).
                    // Premiere ligne : le chemin peut commencer au premier indice ou plus loin dans le second flux.
                    if (r == 0u)
                    {
                        TryCandidate(
                            cd.score, 0.0, Step::None, Step::Diagonal,
                            validCumSumTh, bestCandidate, bestLink, hasCandidate);
                    }
                    else
                    {
                        const Cell* pNW = 0;
                        if (prevRowData)
                        {
                            const unsigned jm1 = jAbs - 1u;
                            if (jm1 >= prevRowStart && jm1 <= prevRowEnd)
                                pNW = &prevRowData[jm1 - prevRowStart];
                        }
                        if (pNW)
                            TryCandidate(
                                cd.score, pNW->cumsum, pNW->link, Step::Diagonal,
                                validCumSumTh, bestCandidate, bestLink, hasCandidate);
                    }

                    // C.6: tester le voisin Ouest (lien horizontal HLink).
                    // Origine : cout precedent 0.
                    if (r == 0u && jAbs == 0u)
                    {
                        TryCandidate(
                            cd.score, 0.0, Step::None, Step::Horizontal,
                            validCumSumTh, bestCandidate, bestLink, hasCandidate);
                    }
                    else if (localJ > 0u)
                    {
                        const Cell& cdW = rowData[localJ - 1u];
                        TryCandidate(
                            cd.score, cdW.cumsum, cdW.link, Step::Horizontal,
                            validCumSumTh, bestCandidate, bestLink, hasCandidate);
                    }

                    // C.7: tester le voisin Nord (lien vertical VLink).
                    // Cas premiere ligne: cout precedent 0.
                    if (r == 0u)
                    {
                        TryCandidate(
                            cd.score, 0.0, Step::None, Step::Vertical,
                            validCumSumTh, bestCandidate, bestLink, hasCandidate);
                    }
                    else
                    {
                        const Cell* pN = 0;
                        if (prevRowData && jAbs >= prevRowStart && jAbs <= prevRowEnd)
                            pN = &prevRowData[jAbs - prevRowStart];
                        if (pN)
                            TryCandidate(
                                cd.score, pN->cumsum, pN->link, Step::Vertical,
                                validCumSumTh, bestCandidate, bestLink, hasCandidate);
                    }

                    // C.8: si aucun voisin valide, la case reste inaccessible.
                    if (!hasCandidate)
                    {
                        cd.cumsum = unreachableCost;
                        cd.link = Step::None;
                        continue;
                    }

                    // C.9: valider la case avec le meilleur cout cumule et le lien choisi.
                    cd.cumsum = bestCandidate;
                    cd.link = bestLink;
                    ++computedInBand;

                    // C.10: mettre a jour le meilleur minimum de la ligne (pour piloter la bande suivante).
                    if (cd.cumsum < validCumSumTh && (!rowBestFound || cd.cumsum < rowBestCost))
                    {
                        rowBestCost = cd.cumsum;
                        rowBestCol = jAbs;
                        rowBestFound = true;
                    }
                }

                if (useStreamingRow && computedInBand > 0u)
                {
                    // Etape 3.4: remettre a jour l'etat de lecture LR apres un calcul en streaming.
                    hasLRLoaded = true;
                    lrLoadedUntil = j - 1u;
                    lrNextExpected = j;
                }

                #ifdef verbose
                if (j <= jEnd)
                    std::cout << "[Dyn] avertissement: lecture LR arretee trop tot a la colonne " << j
                              << " (finBande=" << jEnd << ", ligne=" << r << ")\n";
                #endif

                if (computedInBand == 0u)
                {
                    // Etape 3.5: si rien n'a ete calcule dans la bande, tenter une ligne complete.
                    if (!fallbackFullRow)
                    {
                        #ifdef verbose
                        std::cout << "[Dyn] avertissement: ligne " << r << " vide dans la bande, nouvelle tentative ligne complete\n";
                        #endif
                        fallbackFullRow = true;
                        continue;
                    }
                    #ifdef verbose
                    std::cout << "[Dyn] echec: ligne " << r << " ne contient aucune colonne calculee\n";
                    #endif
                    return false;
                }

                if (!rowBestFound && !fallbackFullRow)
                {
                    // Etape 3.6: si aucun minimum valide dans la bande, retenter sur ligne complete.
                    #ifdef verbose
                    std::cout << "[Dyn] avertissement: ligne " << r << " n'a pas de minimum valide dans la bande, nouvelle tentative ligne complete\n";
                    #endif
                    fallbackFullRow = true;
                    continue;
                }

                if (!rowBestFound)
                {
                    #ifdef verbose
                    std::cout << "[Dyn] echec: ligne " << r << " n'a pas de minimum valide meme en repli\n";
                    #endif
                    return false;
                }
                break;
            }

            // Etape 4: memoriser le meilleur point de la ligne pour piloter la suite.
            bestColRow[r] = rowBestCol;
            bestCostRow[r] = rowBestCost;
            prevBestCol = rowBestCol;
            hasPrevBest = true;

            // Etape 5: adapter dynamiquement la largeur de bande pour la ligne suivante.
            // Adaptation de largeur pour la ligne suivante.
            // 5.1: mesurer la distance du minimum aux deux bords de la bande courante.
            const unsigned distLeft = (rowBestCol >= jStart) ? (rowBestCol - jStart) : 0u;
            const unsigned distRight = (jEnd >= rowBestCol) ? (jEnd - rowBestCol) : 0u;
            const unsigned edgeDist = (distLeft < distRight) ? distLeft : distRight;

            // 5.2: si le minimum est proche d'un bord (ou si on a du faire un full-row fallback),
            // agrandir rapidement la bande pour la ligne suivante.
            if (fallbackFullRow || edgeDist <= edgeGrowMargin)
            {
                unsigned grown = currBandRay + growStep;
                if (grown > hardMaxRay) grown = hardMaxRay;
                currBandRay = grown;
                stableCenteredRows = 0u;
            }

            // 5.3: si le minimum reste bien centre plusieurs lignes de suite,
            // reduire progressivement la bande (sans descendre sous minBandRay).
            else if (edgeDist >= centerStableMargin)
            {
                ++stableCenteredRows;
                if (stableCenteredRows >= stableRowsToShrink && currBandRay > minBandRay)
                {
                    currBandRay = (currBandRay > (minBandRay + shrinkStep)) ? (currBandRay - shrinkStep) : minBandRay;
                    stableCenteredRows = 0u;
                }
            }
            else
            {
                // 5.4: zone intermediaire (ni bord, ni centre stable): on garde le rayon,
                // mais on remet a zero le compteur de stabilite.
                stableCenteredRows = 0u;
            }

            // Etape 6: journaliser la progression globale (meme sans verbose).
            // Log de progression permanent, meme sans verbose.
            if (((r + 1u) % alwaysLogStepRows) == 0u || (r + 1u) == rowCount)
            {
                std::cout << "[Dyn] progression: " << (r + 1u) << "/" << rowCount
                          << " lignes (HR idx=" << iHR << ")\n";
            }
        }

        bool ok = false;
        for (unsigned r = 0; r < (unsigned)Matrice.size(); ++r)
        {
            if (!Matrice[r].empty()) { ok = true; break; }
        }
        #ifdef verbose
        std::cout << "[Dyn] termine ok=" << ok
                      << ", lignes=" << Matrice.size()
                      << ", colsPremiereLigne=" << (Matrice.empty() ? 0 : Matrice[0].size()) << "\n";
        #endif
        return ok;
    }

    // Retro-parcours simple:
    // - choisit le minimum de cumsum sur la ligne cible,
    // - puis remonte en suivant les liens.
    // startHRIdx/endHRIdx: indices HR absolus utilises lors du matching dynamique.
    // Si endHRIdx n'est pas fourni, on demarre de la derniere ligne calculee.
    bool BackTracking(unsigned startHRIdx = 0, unsigned endHRIdx = VideoStream::invalidIndex)
    {
        // Etape 1: verifier que la matrice sparse est disponible et coherente.
        if (Matrice.empty() || matStartLR.size() != Matrice.size()) return false;

        // Etape 2: remettre a zero le marquage du chemin sur toutes les cellules calculees.
        // Reinitialise le marquage du chemin sur les cellules calculees.
        for (unsigned r0 = 0; r0 < (unsigned)Matrice.size(); ++r0)
            for (unsigned c0 = 0; c0 < (unsigned)Matrice[r0].size(); ++c0)
                Matrice[r0][c0].isOnTrack = false;

        // Etape 3: choisir la ligne de depart (derniere ligne, ou ligne correspondant a endHRIdx).
        unsigned startRow = (unsigned)Matrice.size() - 1u;
        if (endHRIdx != VideoStream::invalidIndex)
        {
            if (endHRIdx < startHRIdx) return false;
            startRow = endHRIdx - startHRIdx;
            if (startRow >= (unsigned)Matrice.size()) return false;
        }
        if (Matrice[startRow].empty()) return false;

        // Etape 4: trouver la meilleure colonne de depart sur la ligne cible
        // (minimum de cumsum parmi les cellules valides).
        // Point de depart: minimum de cumsum sur la ligne cible.
        double bestCumSum = unreachableCost;
        unsigned bestCol = 0;
        bool found = false;
        const unsigned c0 = matStartLR[startRow];
        for (unsigned lc = 0; lc < (unsigned)Matrice[startRow].size(); ++lc)
        {
            const double cs = Matrice[startRow][lc].cumsum;
            if (cs >= (unreachableCost * 0.5)) continue; // ignore les valeurs invalides/INF
            if (!found || cs < bestCumSum)
            {
                bestCumSum = cs;
                bestCol = c0 + lc;
                found = true;
            }
        }
        if (!found) return false;

        // Etape 5: remonter le chemin en suivant les liens stockes (D/H/V)
        // et marquer chaque cellule du trajet avec isOnTrack=1.
        // Remontee du chemin en suivant les liens stockes.
        unsigned r = startRow;
        unsigned c = bestCol;
        bool hasTrack = false;
        for (;;)
        {
            Cell* pcd = GetCell(r, c);
            if (!pcd) break;
            Cell& cd = *pcd;
            cd.isOnTrack = true;
            hasTrack = true;

            const Step lk = cd.link;
            if (lk == Step::None) break;

            if (lk == Step::Diagonal)
            {
                if (r == 0 || c == 0) break;
                --r; --c;
                continue;
            }
            if (lk == Step::Horizontal)
            {
                if (c == 0) break;
                --c;
                continue;
            }
            if (lk == Step::Vertical)
            {
                if (r == 0) break;
                --r;
                continue;
            }

            break;
        }

        // Etape 6: retour final (true si au moins une cellule a ete marquee).
        return hasTrack;
    }

    // Exporte 2 parties:
    // 1) Segments extraits du chemin de backtracking
    // 2) Matrice des cumsums ("--" pour valeurs non calculees)
    // startHRIdx: index HR absolu correspondant a la ligne 0 de Matrice.
    bool DumpScoresTxt(const char* fn_out, unsigned startHRIdx = 0, unsigned maxRows = 0) const
    {
        if (!fn_out) return false;
        if (Matrice.empty() || matStartLR.size() != Matrice.size()) return false;
        bool hasAnyCell = false;
        for (unsigned r = 0; r < (unsigned)Matrice.size(); ++r)
        {
            if (!Matrice[r].empty()) { hasAnyCell = true; break; }
        }
        if (!hasAnyCell) return false;

        const unsigned rowsAll = (unsigned)Matrice.size();
        const unsigned rows = (maxRows > 0 && maxRows < rowsAll) ? maxRows : rowsAll;

        auto isOnTrackAt = [&](unsigned r, unsigned c) -> bool
        {
            if (r >= rows) return false;
            const Cell* pcd = GetCell(r, c);
            return pcd && pcd->isOnTrack;
        };
        auto hasTrackedPred = [&](unsigned r, unsigned c) -> bool
        {
            const Cell* pcd = GetCell(r, c);
            if (!pcd) return false;
            const Cell& cd = *pcd;
            const Step lk = cd.link;
            if (lk == Step::Diagonal)
            {
                if (r == 0u || c == 0u) return false;
                return isOnTrackAt(r - 1u, c - 1u);
            }
            if (lk == Step::Horizontal)
            {
                if (c == 0u) return false;
                return isOnTrackAt(r, c - 1u);
            }
            if (lk == Step::Vertical)
            {
                if (r == 0u) return false;
                return isOnTrackAt(r - 1u, c);
            }
            return false;
        };

        // Reconstruit le chemin ordonne start->end.
        std::vector<std::pair<unsigned, unsigned>> path;
        unsigned sr = VideoStream::invalidIndex, sc = VideoStream::invalidIndex;
        bool foundStart = false;
        for (unsigned r = 0; r < rows && !foundStart; ++r)
        {
            const unsigned cStart = matStartLR[r];
            for (unsigned lc = 0; lc < (unsigned)Matrice[r].size(); ++lc)
            {
                const unsigned c = cStart + lc;
                if (!Matrice[r][lc].isOnTrack) continue;
                if (!hasTrackedPred(r, c))
                {
                    sr = r; sc = c;
                    foundStart = true;
                    break;
                }
            }
        }

        if (foundStart)
        {
            unsigned cr = sr, cc = sc;
            path.push_back(std::make_pair(cr, cc));
            for (;;)
            {
                bool moved = false;
                const Cell* pDiag = GetCell(cr + 1u, cc + 1u);
                if (isOnTrackAt(cr + 1u, cc + 1u) && pDiag && pDiag->link == Step::Diagonal)
                {
                    ++cr; ++cc; moved = true;
                }
                else
                {
                    const Cell* pH = GetCell(cr, cc + 1u);
                    if (isOnTrackAt(cr, cc + 1u) && pH && pH->link == Step::Horizontal)
                    {
                        ++cc; moved = true;
                    }
                    else
                    {
                        const Cell* pV = GetCell(cr + 1u, cc);
                        if (isOnTrackAt(cr + 1u, cc) && pV && pV->link == Step::Vertical)
                        {
                            ++cr; moved = true;
                        }
                    }
                }
                if (!moved) break;
                path.push_back(std::make_pair(cr, cc));
            }
        }

        std::ofstream os(fn_out);
        if (!os) return false;

        // =========================
        // 1) SEGMENTS
        // =========================
        os << "=== SEGMENTS ===\n";
        os << "Format:\n";
        os << "[idDebutV1 ; idFinV1] ==> [idDebutV2 ; idFinV2] || [tDebutV1 ; tFinV1] ==> [tDebutV2 ; tFinV2] || nbImg || liaison || dt(diag)\n\n";

        if (path.size() < 2u)
        {
            os << "(aucun segment: chemin absent ou trop court)\n";
            return false;
        }
        else
        {
            auto stepType = [&](unsigned k) -> Step
            {
                const auto& a = path[k - 1u];
                const auto& b = path[k];
                if (b.first == a.first + 1u && b.second == a.second + 1u) return Step::Diagonal;
                if (b.first == a.first && b.second == a.second + 1u) return Step::Horizontal;
                if (b.first == a.first + 1u && b.second == a.second) return Step::Vertical;
                return Step::None;
            };
            auto linkName = [](Step lk) -> const char*
            {
                if (lk == Step::Diagonal) return "High/LowRes";
                if (lk == Step::Horizontal) return "extra LoRes";
                if (lk == Step::Vertical) return "extra HiRes";
                return "inconnue";
            };

            unsigned segIdx = 1u;
            auto emitSegmentAbs = [&](unsigned hr0, unsigned hr1, unsigned lr0, unsigned lr1, Step lk)
            {
                const unsigned th0 = pSrc1->Idx2TimeMs(hr0);
                const unsigned th1 = pSrc1->Idx2TimeMs(hr1);
                const unsigned tl0 = pSrc2->Idx2TimeMs(lr0);
                const unsigned tl1 = pSrc2->Idx2TimeMs(lr1);

                const unsigned dHR = (hr1 >= hr0) ? (hr1 - hr0) : 0u;
                const unsigned dLR = (lr1 >= lr0) ? (lr1 - lr0) : 0u;
                const unsigned nbImg = (dHR > dLR) ? dHR : dLR;

                os << "#" << std::setw(4) << segIdx << " : ";
                os << "[" << hr0 << " ; " << hr1 << "] ==> [" << lr0 << " ; " << lr1 << "]";
                os << " || ";
                os << "[" << th0 << " ; " << th1 << "] ==> [" << tl0 << " ; " << tl1 << "]";
                os << " || " << nbImg;
                os << " || " << linkName(lk);
                if (lk == Step::Diagonal)
                {
                    const int dt0 = (int)th0 - (int)tl0;
                    const int dt1 = (int)th1 - (int)tl1;
                    os << " || dt=[" << dt0 << " ; " << dt1 << "]";
                }
                else
                {
                    os << " || dt=--";
                }
                os << "\n";
                ++segIdx;
            };

            const unsigned hrMin = startHRIdx;
            const unsigned hrMax = startHRIdx + rows - 1u;
            const unsigned lrMin = 0u;
            const unsigned lrMax = (pSrc2->FrameCount() > 0u) ? (pSrc2->FrameCount() - 1u) : 0u;

            // Segment(s) de bord avant le chemin.
            const unsigned hrStart = startHRIdx + path.front().first;
            const unsigned lrStart = path.front().second;
            unsigned leadHR = hrMin, leadLR = lrMin;
            if (hrStart > leadHR)
            {
                emitSegmentAbs(leadHR, hrStart, leadLR, leadLR, Step::Vertical);
                leadHR = hrStart;
            }
            if (lrStart > leadLR)
            {
                emitSegmentAbs(leadHR, leadHR, leadLR, lrStart, Step::Horizontal);
                leadLR = lrStart;
            }

            // Segments internes du chemin (rupture quand le type de lien change).
            unsigned segStart = 0u;
            Step currType = stepType(1u);
            for (unsigned k = 2u; k < (unsigned)path.size(); ++k)
            {
                Step t = stepType(k);
                if (t != currType)
                {
                    const unsigned hr0 = startHRIdx + path[segStart].first;
                    const unsigned lr0 = path[segStart].second;
                    const unsigned hr1 = startHRIdx + path[k - 1u].first;
                    const unsigned lr1 = path[k - 1u].second;
                    emitSegmentAbs(hr0, hr1, lr0, lr1, currType);
                    segStart = k - 1u;
                    currType = t;
                }
            }
            {
                const unsigned hr0 = startHRIdx + path[segStart].first;
                const unsigned lr0 = path[segStart].second;
                const unsigned hr1 = startHRIdx + path.back().first;
                const unsigned lr1 = path.back().second;
                emitSegmentAbs(hr0, hr1, lr0, lr1, currType);
            }

            // Segment(s) de bord apres le chemin.
            unsigned trailHR = startHRIdx + path.back().first;
            unsigned trailLR = path.back().second;
            if (trailLR < lrMax)
            {
                emitSegmentAbs(trailHR, trailHR, trailLR, lrMax, Step::Horizontal);
                trailLR = lrMax;
            }
            if (trailHR < hrMax)
            {
                emitSegmentAbs(trailHR, hrMax, trailLR, trailLR, Step::Vertical);
                trailHR = hrMax;
            }
        }

#ifdef WRITE_MATRICE
        // =========================
        // 2) MATRICE
        // =========================
        os << "\n=== MATRICE (cumsum) ===\n";
        os << "Axe X = indices LR ; Axe Y = indices HR\n";
        os << "Chemin optimal: case entouree de []\n";

        const unsigned cols = matNbLRCols;
        if (cols == 0u) return true;

        os << "HR\\\\LR";
        for (unsigned c = 0; c < cols; ++c) os << '\t' << c;
        os << "\n";

        for (unsigned r = 0; r < rows; ++r)
        {
            const unsigned absHR = startHRIdx + r;
            os << absHR;
            for (unsigned c = 0; c < cols; ++c)
            {
                std::string cell = "--";
                const Cell* pcd = GetCell(r, c);
                if (pcd)
                {
                    const Cell& cd = *pcd;
                    if (cd.cumsum < (unreachableCost * 0.5))
                    {
                        std::ostringstream ss;
                        ss << std::fixed << std::setprecision(4) << cd.cumsum;
                        cell = ss.str();
                    }
                    if (cd.isOnTrack) cell = "[" + cell + "]";
                }
                os << '\t' << cell;
            }
            os << "\n";
        }
#endif
        return true;
    }

    // Exporte la matrice courante dans un fichier texte sous forme de grille.
    // Axe X: VsrcLR, Axe Y: VsrcHR (vertical).
    bool DumpMatriceTxt(const char* fn_out, unsigned maxRows = 0, unsigned maxCols = 0) const
    {
        if (!fn_out) return false;
        if (Matrice.empty() || matStartLR.size() != Matrice.size()) return false;

        unsigned rows = (unsigned)Matrice.size();
        unsigned cols = matNbLRCols;
        if (maxRows > 0 && maxRows < rows) rows = maxRows;
        if (maxCols > 0 && maxCols < cols) cols = maxCols;
        if (rows == 0 || cols == 0) return false;

        std::ofstream os(fn_out);
        if (!os) return false;


        // Format TSV lisible: une ligne par HR, une colonne par LR.
        // Delimiteur tabulation pour ouverture directe dans Excel/LibreOffice.
        os << "# Matrice de cumsum\n";
        os << "# rows=" << rows << "\tcols=" << cols << "\n";
        os << "# chemin optimal: cellule entouree de []\n";
        os << "HR\\\\LR";
        for (unsigned c = 0; c < cols; ++c) os << '\t' << c;
        os << "\n";

        for (unsigned r = 0; r < rows; ++r)
        {
            os << r;
            for (unsigned c = 0; c < cols; ++c)
            {
                const Cell* pcd = GetCell(r, c);
                std::ostringstream ss;
                if (!pcd || pcd->cumsum >= (unreachableCost * 0.5))
                    ss << ".";
                else
                    ss << std::fixed << std::setprecision(4) << pcd->cumsum;

                std::string cellTxt = ss.str();
                if (pcd && pcd->isOnTrack) cellTxt = "[" + cellTxt + "]";
                os << '\t' << cellTxt;
            }
            os << "\n";
        }

        // Copie de la matrice: symboles de lien uniquement.
        // '.' hors chemin, sinon '-' '|' '\' selon le type de lien.
        os << "\n# Matrice des liens\n";
        os << "# '.' hors chemin; '-', '|', '\\' sur chemin\n";
        os << "HR\\\\LR";
        for (unsigned c = 0; c < cols; ++c) os << '\t' << c;
        os << "\n";
        for (unsigned r = 0; r < rows; ++r)
        {
            os << r;
            for (unsigned c = 0; c < cols; ++c)
            {
                char sym = '.';
                const Cell* pcd = GetCell(r, c);
                if (pcd && pcd->isOnTrack)
                {
                    switch (pcd->link)
                    {
                    case Step::Horizontal: sym = '-'; break;
                    case Step::Vertical: sym = '|'; break;
                    case Step::Diagonal: sym = '\\'; break;
                    case Step::None: break;
                    }
                }
                os << '\t' << sym;
            }
            os << "\n";
        }
        return true;
    }
};
