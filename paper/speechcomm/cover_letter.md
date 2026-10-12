Dear Editors of Speech Communication,

Please consider "Thresholded Once, Deployed Elsewhere: A Two-Round Pre-Registered Audit of Threshold Transfer in Speech Deepfake Detection" for publication as a full-length research article.

The manuscript addresses an assessment problem for speech technology: how a countermeasure's decision threshold, chosen on source development speech, behaves when it is applied to other speakers, channels and synthesis systems. Equal error rates and minimum detection costs choose the operating point with the evaluation labels, so they do not show this. We propose a reporting protocol that splits the error at the development threshold into the part that no threshold removes and a threshold transfer gap, reports false rejection and false acceptance under explicit threshold policies, and states the range of thresholds that tie on development data and the speaker-level uncertainty.

We apply the protocol in two separately registered rounds (version-controlled registrations held by the authors; the article reports what had been seen at each step). Round 1 audits five released detectors trained on ASVspoof 2019 LA. Round 2 was registered before any of its scores existed and tests the round-1 observations on three current detectors, eleven retrained AASIST-L runs (with and without silence in training) and eight additional channel conditions. Its registered findings are that the operating policy sets the dominant error for two of three new detectors (the third has a development error rate above the policies' budgets), that the transfer gap recurs across retraining (four of five seeds), and that errors shift toward false rejection in four of eight channel conditions, with low-bitrate codecs and a phone impulse response raising false rejection for every model tested. Exploratory analyses show that removing silence from training moves the operating point, mostly through the weight averaging it switches off, without improving discrimination; that the self-supervised detectors which discriminate In-the-Wild speech well are limited by thresholds their development data do not pin down; and that review budgets designed on development data overflow on shifted targets. We report registered outcomes as registered, including one hypothesis that holds at the clip level but not at the speaker level, which we therefore do not claim.

We believe the article fits the journal's scope under assessment methodologies for speech technology. It builds on established biometric evaluation and calibration practice and comes with a reproducibility package of score files, frozen thresholds, registration snapshots, the deviation log and analysis code.

A separate deployment study, in preparation, covers model conversion, quantization and on-device measurement; none of those results appears in this manuscript. To keep the registration record complete, the manuscript states that the original registration's two quantization hypotheses were not supported, without reporting their analyses.

[Authors: confirm that the manuscript is original, is not under consideration elsewhere and has been approved by all authors; state any related manuscripts or preprints, competing interests, and the public reproducibility URL.]

Sincerely,

[Corresponding author, affiliation and e-mail]
