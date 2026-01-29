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

if !(v"1.12.0-aaa" < VERSION < v"1.13.0")
    error("This script needs to run with Julia 1.12.")
end

import Pkg

project = mktempdir()
cp("./pluto-slider-server-environment", project; force=true)
Pkg.activate(project)
Pkg.add(["URIs", "PlutoPDF"])
Pkg.instantiate()


output_dir = mktempdir(; cleanup=false)
@info "Output directory: $(output_dir)"

lecture_urls = [
    # URLs taken from README lecture table (5 items)
    "https://bertdv.github.io/mlss-2026/lectures/Probability%20Theory%20Review.html",
    "https://bertdv.github.io/mlss-2026/lectures/Bayesian%20Machine%20Learning.html",
    "https://bertdv.github.io/mlss-2026/lectures/Latent%20Variable%20Models%20and%20VB.html",
    "https://bertdv.github.io/mlss-2026/lectures/Intelligent%20Agents%20and%20Active%20Inference.html",
    "https://bertdv.github.io/mlss-2026/lectures/Factor%20Graphs.html",
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

const preamble_html = """
<script>
// Open all <details> on the page
setInterval(() => {
    [...document.querySelectorAll("details")].forEach(detailEl => {
        detailEl.setAttribute("open", true);
    });
}, 1)
</script>
"""

const preamble_html_query = "?preamble_html=$(URIs.escapeuri(preamble_html))"

function output_path(i, url, prefix, base_url_pattern)
    name = URIs.unescapeuri(replace(url, base_url_pattern => "", ".html" => ""))
    output_path = joinpath(output_dir, "$(prefix)$(lpad(i-1, 2, '0')) $(name).pdf")
end

function generate_pdf_collection(urls, prefix, base_url_pattern)
    @info "📚 Processing $(prefix) collection"
    
    # Generate individual PDFs
    for (i, url) in enumerate(urls)
        out_path = output_path(i, url, prefix, base_url_pattern)
        @info "📄 Generating PDF ($(i)/$(length(urls))): $(basename(out_path))"
        PlutoPDF.html_to_pdf(url * preamble_html_query, out_path; open=false, options)
    end
    
    # Merge PDFs
    @info "🗂️ Merging $(prefix) PDFs"
    files = [output_path(i, url, prefix, base_url_pattern) for (i, url) in enumerate(urls)]
    output = joinpath(output_dir, "BMLIP_$(prefix)_Lectures.pdf")
    try
        run(`pdfunite $(files) $output`)
        @info "✅ Output PDF file: $(output)"
    catch
        @error "Failed to merge PDFs" files output
        throw(ErrorException("Failed to merge PDFs"))
    end
    
    return output
end

# Generate B Lectures
generate_pdf_collection(
    lecture_urls,
    "MLSS",
    "https://bertdv.github.io/mlss-2026/lectures/",
)

try
    run(`open $(output_dir)`)
catch
end

