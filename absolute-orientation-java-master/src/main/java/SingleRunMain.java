import de.uni_mannheim.informatik.dws.melt.matching_data.TestCase;
import de.uni_mannheim.informatik.dws.melt.matching_data.Track;
import de.uni_mannheim.informatik.dws.melt.matching_data.TrackRepository;
import de.uni_mannheim.informatik.dws.melt.matching_eval.ExecutionResult;
import de.uni_mannheim.informatik.dws.melt.matching_eval.ExecutionResultSet;
import de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.EvaluatorCSV;
import de.uni_mannheim.informatik.dws.melt.yet_another_alignment_api.Alignment;
import de.uni_mannheim.informatik.dws.melt.yet_another_alignment_api.Correspondence;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * This Main method evaluates a single run.
 * All files must reside in one directory.
 *
 * IMPORTANT: The assumption is that the results-directory contains only a single set of CSV files
 * (i.e., only one configuration).
 */
public class SingleRunMain {


    public static void main(String[] args) {

        String filePath = "C:\\Users\\D060249\\git\\absolute-orientation\\src\\main\\resources\\data\\results_multifarm_0901";
        ExecutionResultSet ers = loadMultifarmAlignmentsFromCsv(filePath);

        // run the evaluation
        EvaluatorCSV evaluatorCSV = new EvaluatorCSV(ers);
        evaluatorCSV.writeToDirectory();
    }


    /**
     * Load an execution result set from the results.
     * @param filePath Path to the multifarm csv directory.
     * @return Execution Result Set instance.
     */
    static ExecutionResultSet loadMultifarmAlignmentsFromCsv(String filePath){

        File resultsDirectory = new File(filePath);

        // map from test case ID to test/train files
        Map<String, File[]> fileMap = new HashMap<>();

        if(!resultsDirectory.exists()){
            System.out.println("The provided directory does not exist. ABORTING PROGRAM.");
            return null;
        }

        for (File file : resultsDirectory.listFiles()) {
            String fileName = file.getName();
            if (fileName.endsWith("_matches_test.csv")) {
                String id = fileName.replaceAll("_matches_test.csv", "");
                fileMap = addEntry(fileMap, id, file);
            } else if (fileName.endsWith("_matches_train.csv")) {
                String id = fileName.replaceAll("_matches_train.csv", "");
                fileMap = addEntry(fileMap, id, file);
            } else {
                System.out.println("Skipping file: " + fileName);
            }
        }

        System.out.println("Files added to map.");

        Track multifarmDeEn = TrackRepository.Multifarm.getSpecificMultifarmTrack("de-en");
        ExecutionResultSet ers = new ExecutionResultSet();
        for (Map.Entry<String, File[]> entry : fileMap.entrySet()) {
            Alignment alignment = new Alignment();
            addToAlignment(entry.getValue()[0], alignment);
            addToAlignment(entry.getValue()[1], alignment);
            TestCase tc = determineTestCase(multifarmDeEn, alignment);
            ers.add(new ExecutionResult(tc, "AbsOrientation", alignment));
        }
        return ers;
    }



    static final String DATASET_PATTERN_STRING = "(?<=^http:\\/\\/).*(?=_)"; // (?<=^http:\/\/).*(?=_)
    static final String LANGUAGE_PATTERN_STRING = "[a-z]{2}(?=#)";
    static Pattern datasetPattern = Pattern.compile(DATASET_PATTERN_STRING);
    static Pattern languagePattern = Pattern.compile(LANGUAGE_PATTERN_STRING);

    static TestCase determineTestCase(Track myTrack, Alignment alignment){
        for (Correspondence c : alignment) {
            String entity1 = c.getEntityOne();
            System.out.println(entity1);
            Matcher datasetMatcher = datasetPattern.matcher(entity1);
            Matcher languageMatcher = languagePattern.matcher(entity1);
            if(!datasetMatcher.find() || !languageMatcher.find()){
                continue;
            }
            String dataset1 = datasetMatcher.group();
            String language1 = languageMatcher.group();

            String entity2 = c.getEntityTwo();
            System.out.println(entity2);
            datasetMatcher = datasetPattern.matcher(entity2);
            languageMatcher = languagePattern.matcher(entity2);
            if(!datasetMatcher.find() || !languageMatcher.find()){
                continue;
            }
            String dataset2 = datasetMatcher.group();
            String language2 = languageMatcher.group();

            String tcName = dataset1 + "-" + dataset2 + "-" + language1 + "-" + language2;
            //System.out.println(tcName);
            return myTrack.getTestCase(tcName);
        }

        return null;
    }

    /**
     * Add the correspondences in the file to the given alignment.
     * @param file The file that is to be added.
     * @param alignment The alignment to which will be added to.
     * @return The alignment to which the addition was performed.
     */
    static Alignment addToAlignment(File file, Alignment alignment) {
        Reader in;
        try {
            in = new FileReader(file);
            CSVFormat format = CSVFormat.Builder.create()
                    .setDelimiter(';')
                    .setSkipHeaderRecord(false) // not working when true, therefore explicitly switching off
                    .build();

            boolean isFirst = true;
            for (CSVRecord record : format.parse(in)){
                if(isFirst){
                    isFirst = false;
                    continue;
                }
                String e1 = "http://" + record.get(0);
                String e2 = "http://" + record.get(2);
                double confidence = Double.parseDouble(record.get(3));
                alignment.add(e1, e2, confidence);
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return alignment;
    }

    private static Map<String, File[]> addEntry(Map<String, File[]> map, String id, File file) {
        if (!map.containsKey(id)) {
            File[] filePair = new File[2];
            filePair[0] = file;
            map.put(id, filePair);
        } else {
            map.get(id)[1] = file;
        }
        return map;
    }


}
