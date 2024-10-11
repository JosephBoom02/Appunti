#let equation_numbering(doc) = {

  set heading(numbering: "1.")
  show heading.where(level: 1): h => {
    h
    // reset equation counter for each chapter
    counter(math.equation).update(0)
  }

  set math.equation(numbering: eqCounter => {
    locate(eqLoc => {
      // numbering of the equation: change this to fit your style
      let eqNumbering = numbering("a", eqCounter)
      // numbering of the heading
      let chapterCounter = counter(heading).at(eqLoc)
      let chapterNumbering = numbering(
        // use custom function to join array into string
        (..nums) => nums
          .pos()
          .map(str)
          .join("."),
        ..chapterCounter
      )
    
      // change this to fit your style
      [(#chapterNumbering - #eqNumbering)]
    })
  })
  
  doc
}