# Reviewer instructions: matched 500k image set

Each of the four reviewers receives one ZIP. All four ZIPs contain the same 64
images in the same order, so their answers can be compared. Work independently
and do not discuss answers until everyone has finished. The model condition,
training settings, generation seed and checker answers are deliberately hidden.

1. Extract your assigned ZIP and open `review.html` in a browser.
2. Open `reviews.csv` in a spreadsheet or text editor. Keep every sample ID
   unchanged and do not add or remove rows or columns.
3. For each image, answer each question with `yes`, `no`, or `unsure`:
   - `target_circle`: Is a circle of the requested color visible?
   - `exactly_two_objects`: Are exactly two distinct recognizable objects visible?
   - `exactly_three_objects`: Are exactly three distinct recognizable objects visible?
4. A distinct object means a separate shape, even if it has the same color as
   another shape. Do not count background speckles. If shapes touch or an
   object cannot be identified or counted confidently, use `unsure` as needed.
5. Save the CSV as CSV, not as an Excel workbook. Return each reviewer's file
   separately; for example, place it at `reviewer-1/reviews.csv` through
   `reviewer-4/reviews.csv` in `outputs/mosaic-matched-review-500k-82000/`.

Every question has its own answer; a circle of the requested color can be
present even when the object count is wrong. Do not inspect the `private`
directory or checker predictions during review. The image set is a small
matched-model evaluation, not the full 300-image checker audit.
