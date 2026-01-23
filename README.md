# MLSS 2026 Melbourne

This site contains the lecture materials for the **Bayesian Machine Learning** lecture (5 February 2026, 11:00 AM) and the **Active Inference** lecture (February 6th, 11:00 AM)

### Instructor

- [Prof.dr. Bert de Vries](http://bertdv.nl) (email: bert.de.vries@tue.nl) 


### Materials

#### Online lecture notes

You can access all lecture materials online through the links in the table below:

<table border = "1">
         <tr>
            <th rowspan = "1"; style="text-align:center">Date</th>
            <th rowspan = "1"; style="text-align:center">lesson</th>
            <th colspan = "1"; style="text-align:center">materials</th>
         </tr>
         <tr>
            <td>5-Feb-2025 <em>(Thu)</em></td>
            <td>1: Probability Theory<br/>
            2: Bayesian Machine Learning</td>
            <td><a href="https://bertdv.github.io/mlss-2026/lectures/Probability%20Theory%20Review.html">PT</a> <br/> 
                <a href="https://bertdv.github.io/mlss-2026/lectures/Bayesian%20Machine%20Learning.html">BML</a></td>
         </tr>
         <tr>
            <td>6-Feb-2026 <em>(Fri)</em></td>
            <td>3: Variational Inference <br/>
            4: Active Inference<br/>
            5: factor graphs (optional)
            </td>
            <td><a href="https://bertdv.github.io/mlss-2026/lectures/Latent%20Variable%20Models%20and%20VB.html">VI</a> <br/> <a href="https://bertdv.github.io/mlss-2026/lectures/Intelligent%20Agents%20and%20Active%20Inference.html">AIF</a>,  <a href="https://github.com/bertdv/mlss-2026/blob/main/lectures/bdv-Feb2026-AIF-lecture.ppsx">AIF slides</a><br/>
            <a href="https://bertdv.github.io/mlss-2026/lectures/Factor%20Graphs.html">FFG</a>     
            </td>
         </tr>
      </table>

#### Full course

The course materials are derived from a full MSc-level course on [Bayesian Machine Learning and Information Processing](http://bmlip.nl), which is taught annually at [TU Eindhoven](https://www.tue.nl/en/) by the same instructor. For additional background and content, please refer to the materials of that course.

#### PDF notes

If necessary, you can download the lecture notes in PDF format here:

- [lecture notes PDF](https://github.com/bertdv/mlss-2026/blob/main/lectures/pdf/BMLIP_MLSS_Lectures.pdf) version 21-Jan-2026.

However, we recommend that you read the lecture notes in your browser to take advantage of the interactive materials that we prepared for this course, based on [Pluto.jl](https://plutojl.org/).


<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>






### BELOW HERE ONLY FOR INSTRUCTOR

#### How to generate a new PDF bundle for the lecture notes?

- brew install `poppler`
- Open a terminal in the `mlss-2026` folder
- Run `julia tools/generate_pdf.jl` in the root folder

This gives an output PDF file. Then you should:

1. Go to https://github.com/bertdv/mlss-2026/releases and make a new release (mlss2, mlss3, etc)
2. Attach the PDF as "Binary" of the release, and publish
3. Take the new PDF URL, and write it in README.md

