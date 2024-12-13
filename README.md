# NetworkTrace

This is a collection of two Python scripts I wrote one morning as part of my day job, and a CSV of the tabular output; a visualization of the output can be found at https://www.youtube.com/shorts/Q4WeYX0ryKU.

The origin of this was a consequence/likelihood-of-failure analysis I was performing for a utility client in Florida, as part of an effort to prioritize their watermains for upgrades. Normal parameters we have in the analysis (things like pipe age, material, etc.) were sparse in their data, so we were looking for some additional metrics by which we could further evaluate their system. I came up with the idea of "cumulative distance from source" and "downstream customer count" - i.e., let's trace the pipe network to see how far each of the ~75,000 pipe segments (polylines) were from the utility's two water treatment facility and see how much of the customer base would be SOL if a specific segment of pipe had a break; the former is doable from hydraulic models (which we didn't have) but the latter is not very easily, so it was a unique approach the client liked. Not as accurate a full hydraulic model due to loops, but enough for a quick & fast analysis that could drive further decision-making.

I also wanted it to kick out a visual so that we could QA the script and also show something "cool" to the client other than just a very large table of nunmbers.

This uses open-source code only, specifically geopandas to do the geospatial ops. In each iteration, we start with a shortlist of pipelines whose downstream distances are defined (at the treatment plants, this is the ones that surround it). We parallel-process each of those defined pipes, using geopandas to find pipes with undefined downstream distances that intersect the target pipe. We then define the downstream distance for the intersecting pipes, and bring them back to an array to be fed into the next iteration. We eventually get every pipe that geopsatially "touches" the network - there are some disconnected pipes due to errors in the client's source datafiles as well as some interconnections through pipe they don't own, but it was successful at processing the network in about 3 hours (354 iterations) on my local machine (was not allowed to move data to the cloud for remote processing due to security concerns regarding public water systems), each iteration being limited by the number of connected pipes. Not the fastest algorithm, but effective.

There's also a second script that leverages the output from the network trace and a database of customer meters to get the second parameter (# of affected downstream customers) for the analysis.
