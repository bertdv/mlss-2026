if !isdir("pluto-slider-server-environment")
    error("""
    Run me from the root of the repository directory, using:

    julia tools/generate_pdf.jl
    """)
end

try
    @assert read(`pdfunite -v`, String) isa String
catch
    error("pdfunite is not installed. Please install it using your package manager. This is used to merge the individual PDFs into a single PDF.")
end

# if !(v"1.12.0-aaa" < VERSION < v"1.13.0")
#     error("Our notebook package environments need to be updated with Julia 1.11. Go to julialang.org/downloads to install it.")
# end

import Pkg

project = mktempdir()
cp("./pluto-slider-server-environment", project; force=true)
Pkg.activate(project)
Pkg.add(["URIs", "PlutoPDF"])
Pkg.instantiate()



lectures_dir = mktempdir(; cleanup=false)

lecture_urls = [
    "https://bmlip.github.io/course/lectures/Course%20Syllabus.html",
    "https://bmlip.github.io/course/lectures/Machine%20Learning%20Overview.html",
    "https://bmlip.github.io/course/lectures/Probability%20Theory%20Review.html",
    "https://bmlip.github.io/course/lectures/Bayesian%20Machine%20Learning.html",
    "https://bmlip.github.io/course/lectures/Factor%20Graphs.html",
    "https://bmlip.github.io/course/lectures/The%20Gaussian%20Distribution.html",
    "https://bmlip.github.io/course/lectures/The%20Multinomial%20Distribution.html",
    "https://bmlip.github.io/course/lectures/Regression.html",
    "https://bmlip.github.io/course/lectures/Generative%20Classification.html",
    "https://bmlip.github.io/course/lectures/Discriminative%20Classification.html",
    "https://bmlip.github.io/course/lectures/Latent%20Variable%20Models%20and%20VB.html",
    "https://bmlip.github.io/course/lectures/Dynamic%20Models.html",
    "https://bmlip.github.io/course/lectures/Intelligent%20Agents%20and%20Active%20Inference.html"
]

prop_prog_urls = [
    "https://bmlip.github.io/course/probprog/PP0%20-%20Introduction%20Bayesian%20Machine%20Learning.html",
    "https://bmlip.github.io/course/probprog/PP1%20-%20Bayesian%20inference%20in%20conjugate%20models.html",
    "https://bmlip.github.io/course/probprog/PP2%20-%20Bayesian%20regression%20and%20classification.html",
    "https://bmlip.github.io/course/probprog/PP3%20-%20variational%20Bayesian%20inference.html",
    "https://bmlip.github.io/course/probprog/PP4%20-%20Bayesian%20filtering%20and%20smoothing.html",
]

import PlutoPDF
import URIs


const options = (
    format ="A4",
    margin=(
        top="30mm",
        right="15mm",
        bottom="30mm",
        left="10mm",
    ),
    printBackground=true,
    displayHeaderFooter=false,
)

function output_path(i, url)
    name = URIs.unescapeuri(replace(url, "https://bmlip.github.io/course/lectures/" => "", ".html" => ""))
    output_path = joinpath(lectures_dir, "B$(lpad(i-1, 2, '0')) $(name).pdf")
end

for (i,url) in enumerate(lecture_urls)
    out_path = output_path(i, url)
    
    @info "📄 Generating lecture PDF ($(i)/$(length(lecture_urls)))" out_path
    PlutoPDF.html_to_pdf(url, out_path; open=false, options)
end

@info "🗂️📚 Merging lecture PDFs"

files = [
    output_path(i, url) for (i, url) in enumerate(lecture_urls)
]

output = joinpath(lectures_dir, "BMLIP Lectures.pdf")

run(`pdfunite $(files) $output`)


@info "✅ Output pdf file:" output

