# Crop table: sources, what they support, and what is still judgment

Last reviewed: 14 Nov 2026. Values in `crops.csv` were cross-checked against the sources below.
Confidence reflects how directly the sources support the number, not how important it is.

## How `water_mm` is defined (important)
Seasonal crop water demand in mm. For **all crops except rice** this is crop evapotranspiration (what the plant uses,
mostly FAO ranges). For **rice** it is the *total field water including puddling and percolation losses*, because
FAO's rice figure (450-700 mm) covers crop use only and badly understates what paddy actually receives in Punjab.
Consequence: irrigation needs for non-rice crops are probably a little understated, since real irrigation also loses
water in canals and fields. The model does not apply an irrigation-efficiency factor.

## Values and evidence
| Crop | water_mm used | Evidence found | Duration used | Evidence found |
|---|---|---|---|---|
| Wheat | 450 | FAO range 450-650 (S1). Indian Punjab total ~400 (S2); CROPWAT study 411 mm | 140 | FAO 120-150 days (S1) |
| Rice | 1200 | Total rice water 1144-1560 mm across Indo-Gangetic plains (S2); farmer practice often 1500-3000 mm of irrigation in NW India (S3) | 130 | FAO 90-150 days (S1) |
| Cotton | 900 | Cotton Research Institute Multan: 800-1000 mm (S4); FAO 700-1300 (S1) | 180 | FAO 180-195 days (S1) |
| Maize | 600 | Pakistan sources 500-750 mm (S8); FAO 500-800 (S1) | 115 | Full-season variety ~115 days (S8); autumn sowing late Jul-mid Aug |
| Sorghum fodder | 400 | FAO sorghum 450-650 for a 120-130 day grain crop (S1); scaled down for a ~90 day fodder cycle | 90 | First cut ~60 days, second ~45 days later (S13) |
| Chickpea | 250 | Consumptive use ~210-216 mm in a central India trial (S9); mostly rainfed in Pakistan | 150 | 130-160 days (S9) |
| Mustard | 350 | Ranges 190-400 mm and 400-600 mm in extension sources (S11) | 135 | ~130 days for short canola types (S11) |
| Berseem | 700 | Consumptive use 594-774 mm in northern Punjab, Pakistan (S10); 700-800 mm in Indian extension material | 150 | Sown Sep-Nov, cut until Mar-Apr (S10) |
| Mungbean | 300 | No direct figure found. Consistent with FAO bean 300-500 (S1) and 2-4 irrigations (S12). **Low confidence.** | 65 | 55-70 day short-duration varieties (S12) |

## Corrections made to my earlier draft
- Wheat water 400 -> 450, cotton 700 -> 900, maize 500 -> 600, berseem 600 -> 700, mustard 300 -> 350 (earlier values sat at or below the low end of the published ranges).
- Durations: chickpea 130 -> 150, mustard 125 -> 135, maize 110 -> 115, mungbean 70 -> 65.
- Chickpea frost tolerance 3 -> 2 (frost at flowering causes seed failure, S9).
- Cotton income 4 -> 3 (recent Pakistani and Indian Punjab studies describe cotton returns as lower and more volatile, S14).
- Cotton note changed from "whitefly" to leaf curl virus and pink bollworm, which the sources actually name (S4, S5).
- Added: rice water definition (see above), prussic acid warning for sorghum, maize waterlogging and pollen limits.

## Still judgment, not sourced (replace with local data if you can)
- All 1-5 ratings for `root`, `residue`, `heat_tol`, `frost_tol`, `drought_tol`. The sources support the direction
  (e.g. sorghum drought/heat tolerant, maize drought-sensitive, chickpea waterlogging-sensitive, wheat terminal-heat sensitive)
  but not the exact numbers.
- `income` (1-5). No clean, current, comparable per-acre net-return table for Pakistan's Punjab was found.
  Returns change a lot with prices and year. Treat this as a placeholder, say so, and consider replacing it with
  Punjab Agriculture Department cost-of-production figures.
- `n_fixer` and `family` are standard botany and are reliable.

## Source list
Quality: A = official/peer-reviewed/institutional, B = extension or news reporting, C = course notes or blogs (weakest; use only to support direction).

- **S1** (A) FAO, Irrigation Water Management training manual, Ch. 2, Tables 4 and 5 (seasonal water needs and growing periods): https://www.fao.org/3/s2022e/s2022e02.htm
- **S2** (A) ICAR, Water management in rice-wheat cropping system in Indo-Gangetic Plains (rice 1144-1560 mm, wheat up to 400 mm): https://epubs.icar.org.in/index.php/IJEE/article/view/125672
- **S3** (A) Sandhu et al., Punjab Agricultural University, irrigation scheduling for rice (1500-3000 mm irrigation under common practice): https://core.ac.uk/download/158352881.pdf
- **S4** (A) Central Cotton Research Institute, Multan, cotton production technology (800-1000 mm): https://ccri.gov.pk/tt.html
- **S5** (B) Dawn, cotton sowing, leaf curl virus and wheat timing: https://www.dawn.com/news/721959 and https://www.dawn.com/news/74131
- **S6** (B) APP/Punjab Agriculture Department, wheat sowing window Nov 1-20 and ~1% yield loss per day after: https://www.app.com.pk/?p=1110857
- **S7** (B) Tribune India reporting PAU advice (sow by Nov 15; terminal heat stress): https://www.tribuneindia.com/news/punjab/sow-wheat-by-nov-15-to-prevent-grain-shrivelling-pau-to-farmers-448039
- **S8** (B/C) Maize in Pakistan: https://www.dawn.com/news/236949 and https://technologytimes.pk/?p=48888 (500-750 mm, 115-day varieties, pollen and waterlogging limits)
- **S9** (A/B/C) Chickpea: PAU package of practices https://www.vikaspedia.in/agriculture/crop-production/package-of-practices/pulses/chick-pea ; consumptive-use trial https://pub.isa-india.in/index.php/ija/article/download/3983/3668 ; Pakistan production guide (blog, weakest) https://mazarieducation.blogspot.com/2026/02/production-technology-of-chickpea-gram.html ; drought tolerance reporting https://dawn.com/news/1310061
- **S10** (A/B) Berseem: consumptive use 594-774 mm, northern Punjab, Pakistan https://tehqeeqat.org/urdu/articleDetails/36108 ; irrigations and cuttings https://www.dawn.com/news/274300 ; sowing and last-cut dates (Pakistan Journal of Agricultural Research) https://mail.researcherslinks.com/current-issues/Effect-Last-Fodder-Cut-on-Fodder-Seed-Yield-Berseem/24/1/3876/html
- **S11** (A/B/C) Mustard: Punjab Agriculture Department varieties https://agripunjab.punjab.gov.pk/node/9623 ; Pakissan crop guide https://www.pakissan.com/english/allabout/crop/rapeseed.shtml ; water-requirement ranges from Indian extension notes (weakest) https://www.slideshare.net/slideshow/lecture-6-rapeseed-and-mustard-related-topics/265053241
- **S12** (A/B) Mungbean: https://mail.pakbs.org/pjbot/PDFs/42(5)/PJB42(5)3125.pdf (55-70 day varieties, Pakistan) ; https://kissan.root.pk/agricultural-library/70-moong
- **S13** (B/C) Sorghum fodder: https://www.cropandweed.com/archives/2023/vol19issue2/19-2-20.pdf (cutting schedule) ; https://core.ac.uk/download/357559386.pdf (drought and heat tolerance)
- **S14** (B) Business Recorder, competing crops, productivity vs profitability: https://www.brecorder.com/news/40183742 ; ICAR study of Punjab (India) crop sequences https://krishi.icar.gov.in/jspui/bitstream/123456789/24310/1/Crop%20performance%20in%20Punjab_raju%20et%20al.pdf

Note: several sources are from Indian Punjab (similar climate and cropping system) because they were the most detailed;
Pakistani sources were used wherever available. Check local figures with your district extension office.
