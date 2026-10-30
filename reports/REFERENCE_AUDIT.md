# Reference audit

Every entry in the bibliography was checked against a primary or
institutional source. Nothing below is taken from recollection: each row
names the source that was actually read.

Two entries were misattributed, two could not be located at all, and three
carried volume/issue/page data that no source confirmed.

## Summary

| # | Status | Action |
|---|---|---|
| [1] Van Do et al., HPA properties | Partially verified | Venue unconfirmed; now cited as a preprint |
| [2] Jeong & Jeong, ARAScaler | **Verified in full** | DOI added |
| [3] Fog et al., SOSE 2025 | **Verified in full** | DOI added |
| [4] "Combined vertical and horizontal autoscaling through MPC" | **Misattributed** | Authors corrected |
| [5] Quattrocchi et al., TSC 2024 | **Verified in full** | DOI added |
| [6] OptScaler | **Misattributed** | Authors corrected, DOI added |
| [7] Cai et al., TSC 2025 | Existence corroborated | Reduced to "Z. Cai et al." |
| [8] Meng et al., CATScaler | Existence verified | Unconfirmed vol/pages removed |
| [9] Ding et al., Informer autoscaling | **Not located** | **Removed** |
| [10] Fang et al., HGraphScale | Authors verified | Unconfirmed vol/pages removed |
| [11] Habibi et al., AgileSphere | **Not located** | **Removed** |
| [12] Kumar, Verma & Buyya, arXiv:2606.07046 | **Verified in full** | Retained |

## The two misattributions

### Reference [6] — OptScaler

The manuscript credited this to "D. Zhang et al." The version of record
credits **Ding Zou, Wei Lu, Zhibo Zhu, Xingyu Lu, Jun Zhou, Xiaojin Wang,
Kangyu Liu, Kefan Wang, Renen Sun, Haiqing Wang**.

- Source: the published PDF, `vldb.org/pvldb/vol17/p4090-lu.pdf`, title page
  and PVLDB reference line.
- PVLDB **17**(12):4090–4103, 2024. DOI `10.14778/3685800.3685829`.
- Note: the VLDB site's own metadata page lists the first author as "Aaron
  Zou". The printed paper and the arXiv record both say "Ding Zou", so the
  printed form is used.

The in-text mention in Section II-C was corrected from "Zhang et al. (2024)"
to "Zou et al. (2024)".

### Reference [4] — Euro-Par 2018 MPC autoscaling

The manuscript credited "Combined vertical and horizontal autoscaling
through model predictive control" to **L. Baresi, S. Guinea, A. Leva and
G. Quattrocchi**. It is by **E. Incerto, M. Tribastone and C. Trubiani**.

- Sources: two independent institutional repositories, IMT Lucca
  (`iris.imtlucca.it/handle/20.500.11771/16499`) and GSSI
  (`iris.gssi.it/handle/20.500.12571/7034`), which agree on the author list,
  the page range 147–159 and the DOI.
- Euro-Par 2018, LNCS **11014**, pp. 147–159. DOI
  `10.1007/978-3-319-96983-1_11`.

This one matters beyond the byline: [4] is cited at eight places and the
literature table credits it with "first MPC formulation for container
elasticity", so the paper's account of who established that line of work was
wrong. The characterisation itself holds — the title is literally the claim —
so only the attribution changed. The in-text mention in Section II-C was
corrected to "Incerto et al. (2018)".

## The two removals

Both were searched twice, including an extended search, by exact title and by
author-plus-topic.

**[9] Y. Ding, C. Li, Z. Cai, X. Wang, B. Yang, "A dynamic interval
auto-scaling optimization method based on Informer time series prediction,"
IEEE Access 13:14572–14583, 2025.** No record found under this title. An
Informer-based autoscaler is discussed in the 2026 survey [12], but under a
different name ("InformerAutoScale"), and that does not establish this title,
author list, volume or page range.

**[11] P. Habibi, S. Rawat, A. Leon-Garcia, "AgileSphere:
Predictive-reactive Kubernetes cluster autoscaling with warm-pool node
promotion," IEEE Access 14:129559–129579, 2026.** No record found for
"AgileSphere" in any form.

Both were removed rather than retained with a caveat, and the remaining ten
references were renumbered ([10]→[9], [12]→[10]). The literature table lost
the corresponding two rows, and the one in-text citation group that named [9]
was corrected.

These two originated in the reference list supplied at the start of the
project rather than from this survey. Their presence is a reason to treat the
rest of that list as unverified until checked, which is what this audit did.

## Unconfirmed bibliographic data removed

Three entries exist but carry volume/issue/page data that no accessible
source confirmed. Rather than print unverified numbers, the citations now
stop at the journal and year:

- **[7] Cai et al.** — existence and topic corroborated by the body text of
  [12] ("Cai et al. [35] introduced a deep learning and feedback control
  based autoscaling framework for cloud-native microservices"). Co-author
  list unconfirmed, so reduced to "Z. Cai et al."
- **[8] Meng et al., CATScaler** — exists (IEEE Xplore document 11095836,
  found by exact-title search); first author corroborated by [12] ("Meng et
  al. [36] proposed CATScaler"). The claimed "vol. 18, no. 5, pp. 2659–2672"
  could not be confirmed, so it was dropped; reduced to "F. Meng et al."
- **[10] Fang, Ma, Chen, Buyya, HGraphScale** — author list verified from the
  accepted manuscript hosted by Victoria University of Wellington. That PDF
  is the LaTeX-template version with unfilled "VOL. X, NO. X, X 202X"
  placeholders, so the claimed "vol. 19, no. 1, pp. 410–422, 2026" is
  unsupported and was dropped; the year is given as 2025, matching the
  manuscript's own filename and its arXiv listing date.

## Reference [1] — venue unconfirmed

"Properties of horizontal pod autoscaling algorithms and application for
scaling cloud-native network functions" exists, and the body text of [12]
corroborates "Van Do et al." as the authors of an HPA-properties study. The
only locatable copy sits on a preprint host whose robots policy blocked
retrieval, so the venue could not be confirmed. The entry had no venue to
begin with; it is now explicitly marked "preprint" rather than left looking
like a journal paper with a missing field.

## What this audit is not

This is source verification, not a similarity check. It says nothing about
textual overlap with published work. No Turnitin or iThenticate report was
produced, and none of the statements here should be read as a plagiarism
percentage.
