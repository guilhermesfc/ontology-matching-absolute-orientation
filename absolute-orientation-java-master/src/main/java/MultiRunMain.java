import de.uni_mannheim.informatik.dws.melt.matching_data.GoldStandardCompleteness;
import de.uni_mannheim.informatik.dws.melt.matching_data.TestCase;
import de.uni_mannheim.informatik.dws.melt.matching_data.Track;
import de.uni_mannheim.informatik.dws.melt.matching_data.TrackRepository;
import de.uni_mannheim.informatik.dws.melt.matching_eval.ExecutionResult;
import de.uni_mannheim.informatik.dws.melt.matching_eval.ExecutionResultSet;
import de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.EvaluatorCSV;
import de.uni_mannheim.informatik.dws.melt.matching_eval.paramtuning.ConfidenceFinder;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.filter.ConfidenceFilter;
import de.uni_mannheim.informatik.dws.melt.yet_another_alignment_api.Alignment;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.jena.ontology.OntModel;

import java.io.*;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

/**
 * This Main method evaluates a single run.
 * All files must reside in one directory.
 *
 * IMPORTANT: The result directory may contain multiple configurations with regards to the training percentage.
 */
public class MultiRunMain {


    public static void main(String[] args) {
        String filePath = "C:\\Users\\D060249\\git\\absolute-orientation\\src\\main\\resources\\data\\final";
        ExecutionResultSet ers = loadMultifarmAlignmentsFromCsv(filePath);

        //ers.addAll(applyOptimalThresholding(ers));
        EvaluatorCSV evaluatorCSV = new EvaluatorCSV(ers);

        evaluatorCSV.writeToDirectory();
    }


    static ExecutionResultSet applyOptimalThresholding(ExecutionResultSet ers){
        ExecutionResultSet result = new ExecutionResultSet();
        ConfidenceFilter filter = new ConfidenceFilter();
        for (ExecutionResult er : ers) {
            // latest build version with precision fix
            //double optimalThreshold = ConfidenceFinder.getBestConfidenceForFmeasure(er.getReferenceAlignment(), er.getSystemAlignment(), GoldStandardCompleteness.COMPLETE, 100);

            // latest release version
            double optimalThreshold = ConfidenceFinder.getBestConfidenceForFmeasure(er.getReferenceAlignment(), er.getSystemAlignment(), GoldStandardCompleteness.COMPLETE);
            filter.setThreshold(optimalThreshold);
            Alignment refinedAlignment = filter.filter(
                    er.getSystemAlignment(),
                    er.getSourceOntology(OntModel.class),
                    er.getTargetOntology(OntModel.class));
            result.add(new ExecutionResult(
                    er.getTestCase(),
                    er.getMatcherName() + "_optimal_threshold",
                    refinedAlignment));
        }
        return result;
    }


    /**
     * Load an execution result set from the results.
     * @param filePath Path to the multifarm csv directory.
     * @return Execution Result Set instance.
     */
    static ExecutionResultSet loadMultifarmAlignmentsFromCsv(String filePath){
        File resultsDirectory = new File(filePath);

        // map from test case ID to sampling rate to test/train files
        Map<TestCase, Map<Double, Alignment>> fileMap = new HashMap<>();

        if (!resultsDirectory.exists()) {
            System.out.println("The provided directory does nto exist. ABORTING PROGRAM.");
            return null;
        }

        for (File file : resultsDirectory.listFiles()) {
            String fileName = file.getName();
            if (fileName.endsWith("_matches_test.csv") || fileName.endsWith("_matches_train.csv")) {
                // do nothing
            } else if (fileName.endsWith(".csv")) {

                try (Reader reader = new FileReader(file)) {

                    Iterable<CSVRecord> parser = CSVFormat.Builder.create()
                            .setDelimiter(",")
                            .build().parse(reader);

                    boolean firstRun = true;
                    for (CSVRecord r : parser) {
                        if (firstRun) {
                            firstRun = false;
                            continue;
                        }
                        Double rate = Double.parseDouble(r.get(15));
                        String fileId = file.getName().substring(0, file.getName().length() - 4);
                        File test_file = new File(file.getParentFile(), fileId + "_matches_test.csv");
                        File train_file = new File(file.getParentFile(), fileId + "_matches_train.csv");

                        if (!test_file.exists()) {
                            System.out.println("ERROR: The following file does not exist: " + test_file.getAbsolutePath());
                            break;
                        }
                        if (!train_file.exists()) {
                            System.out.println("ERROR: The following file does not exist: " + train_file.getAbsolutePath());
                            break;
                        }

                        Alignment alignment = new Alignment();
                        SingleRunMain.addToAlignment(test_file, alignment);
                        SingleRunMain.addToAlignment(train_file, alignment);

                        TestCase tc = SingleRunMain.determineTestCase(
                                TrackRepository.Multifarm.getSpecificMultifarmTrack("de-en"),
                                alignment);
                        
                        if (tc == null) {
                            System.out.println("Could not determine test case: "+test_file);
                            continue;
                        }

                        // now we add
                        if (fileMap.containsKey(tc)) {
                            Map<Double, Alignment> rateAlignment = fileMap.get(tc);
                            rateAlignment.put(rate, alignment);
                        } else {
                            HashMap rateAlignment = new HashMap();
                            rateAlignment.put(rate, alignment);
                            fileMap.put(tc, rateAlignment);
                        }
                        break;
                    }

                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }

            } else {
                // We should never arrive here...
                System.out.println("ERROR: Skipping file: " + fileName);
            }
        } // end of loop over files


        // now let's evaluate
        ExecutionResultSet ers = new ExecutionResultSet();
        for (Map.Entry<TestCase, Map<Double, Alignment>> entry : fileMap.entrySet()) {

            for (Map.Entry<Double, Alignment> rateAlignmentMap : entry.getValue().entrySet()) {
                ers.add(new ExecutionResult(
                        entry.getKey(),
                        "M_" + rateAlignmentMap.getKey(),
                        rateAlignmentMap.getValue()
                ));
            }
        }
        return ers;
    }


}



