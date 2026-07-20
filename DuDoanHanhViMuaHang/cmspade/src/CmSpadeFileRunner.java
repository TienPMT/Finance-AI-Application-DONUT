import algorithms.AlgoCMSPADE;
import algorithms.candidatePatternsGeneration.CandidateGenerator;
import algorithms.candidatePatternsGeneration.CandidateGenerator_Qualitative;
import algorithms.dataStructures.creators.AbstractionCreator;
import algorithms.dataStructures.creators.AbstractionCreator_Qualitative;
import algorithms.dataStructures.database.SequenceDatabase;
import algorithms.idLists.creators.IdListCreator;
import algorithms.idLists.creators.IdListCreator_FatBitmap;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class CmSpadeFileRunner {
    public static void main(String[] args) throws IOException {
        RunnerOptions options = RunnerOptions.parse(args);
        Path sanitizedInputPath = sanitizeSequenceDb(options.inputPath);

        AbstractionCreator abstractionCreator = AbstractionCreator_Qualitative.getInstance();
        IdListCreator idListCreator = IdListCreator_FatBitmap.getInstance();
        CandidateGenerator candidateGenerator = CandidateGenerator_Qualitative.getInstance();
        SequenceDatabase sequenceDatabase = new SequenceDatabase(abstractionCreator, idListCreator);

        sequenceDatabase.loadFile(sanitizedInputPath.toString(), options.minSupport);
        if (sequenceDatabase.size() == 0) {
            throw new IllegalArgumentException("sequence DB has no valid sequences");
        }

        AlgoCMSPADE algo = new AlgoCMSPADE(options.minSupport, true, abstractionCreator);
        algo.runAlgorithm(
                sequenceDatabase,
                candidateGenerator,
                true,
                options.verbose,
                options.outputPath.toString(),
                false
        );

        System.out.println("sequence_count=" + sequenceDatabase.size());
        System.out.println("pattern_count=" + algo.getNumberOfFrequentPatterns());
        System.out.println("minsup=" + options.minSupport);
        System.out.println(algo.printStatistics());
    }

    private static Path sanitizeSequenceDb(Path inputPath) throws IOException {
        Path sanitizedPath = Files.createTempFile("cmspade-sequence-db-", ".txt");
        sanitizedPath.toFile().deleteOnExit();
        boolean hasSequence = false;
        StringBuilder sanitized = new StringBuilder();
        try (BufferedReader reader = Files.newBufferedReader(inputPath)) {
            String line;
            int lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#") || trimmed.startsWith("%") || trimmed.startsWith("@")) {
                    continue;
                }
                hasSequence = true;
                validateSequenceLine(trimmed.split("\\s+"), lineNumber);
                sanitized.append(trimmed).append(System.lineSeparator());
            }
        }
        if (!hasSequence) {
            throw new IllegalArgumentException("sequence DB has no valid sequences");
        }
        Files.writeString(sanitizedPath, sanitized.toString());
        return sanitizedPath;
    }

    private static void validateSequenceLine(String[] tokens, int lineNumber) {
        if (tokens.length < 3 || !tokens[tokens.length - 1].equals("-2")) {
            throw invalidGrammar(lineNumber, "line must end with -2 after at least one closed itemset");
        }
        int endCount = 0;
        boolean hasOpenItemset = false;
        boolean hasItemInCurrentItemset = false;
        boolean hasClosedItemset = false;
        for (int index = 0; index < tokens.length; index++) {
            String token = tokens[index];
            validateToken(token, lineNumber);
            if (token.equals("-2")) {
                endCount++;
                if (index != tokens.length - 1) {
                    throw invalidGrammar(lineNumber, "tokens found after -2");
                }
                if (hasOpenItemset) {
                    throw invalidGrammar(lineNumber, "itemset must close with -1 before -2");
                }
                continue;
            }
            if (token.equals("-1")) {
                if (!hasOpenItemset || !hasItemInCurrentItemset) {
                    throw invalidGrammar(lineNumber, "empty itemset is not allowed");
                }
                hasOpenItemset = false;
                hasItemInCurrentItemset = false;
                hasClosedItemset = true;
                continue;
            }
            hasOpenItemset = true;
            hasItemInCurrentItemset = true;
        }
        if (endCount != 1 || !hasClosedItemset) {
            throw invalidGrammar(lineNumber, "line must contain exactly one -2 and at least one itemset");
        }
    }

    private static void validateToken(String token, int lineNumber) {
        if (token.equals("-1") || token.equals("-2")) {
            return;
        }
        if (token.startsWith("<") && token.endsWith(">")) {
            parseInteger(token.substring(1, token.length() - 1), token, lineNumber);
            return;
        }
        int value = parseInteger(token, token, lineNumber);
        if (value < 0) {
            throw new IllegalArgumentException("invalid token '" + token + "' at line " + lineNumber);
        }
    }

    private static IllegalArgumentException invalidGrammar(int lineNumber, String reason) {
        return new IllegalArgumentException("invalid sequence grammar at line " + lineNumber + ": " + reason);
    }

    private static int parseInteger(String value, String token, int lineNumber) {
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException("invalid token '" + token + "' at line " + lineNumber, error);
        }
    }

    private static class RunnerOptions {
        private final Path inputPath;
        private final Path outputPath;
        private final double minSupport;
        private final boolean verbose;

        private RunnerOptions(Path inputPath, Path outputPath, double minSupport, boolean verbose) {
            this.inputPath = inputPath;
            this.outputPath = outputPath;
            this.minSupport = minSupport;
            this.verbose = verbose;
        }

        private static RunnerOptions parse(String[] args) {
            Path inputPath = null;
            Path outputPath = null;
            Double minSupport = null;
            boolean verbose = false;

            for (int index = 0; index < args.length; index++) {
                String arg = args[index];
                if (arg.equals("--input")) {
                    inputPath = Path.of(readValue(args, ++index, arg)).toAbsolutePath().normalize();
                } else if (arg.equals("--output")) {
                    outputPath = Path.of(readValue(args, ++index, arg)).toAbsolutePath().normalize();
                } else if (arg.equals("--minsup")) {
                    minSupport = Double.parseDouble(readValue(args, ++index, arg));
                } else if (arg.equals("--verbose")) {
                    verbose = true;
                } else {
                    throw new IllegalArgumentException("unknown argument: " + arg);
                }
            }

            if (inputPath == null || outputPath == null || minSupport == null) {
                throw new IllegalArgumentException("usage: --input PATH --output PATH --minsup VALUE [--verbose]");
            }
            if (!Double.isFinite(minSupport) || minSupport <= 0.0 || minSupport > 1.0) {
                throw new IllegalArgumentException("minsup must be in (0, 1]");
            }
            return new RunnerOptions(inputPath, outputPath, minSupport, verbose);
        }

        private static String readValue(String[] args, int index, String flag) {
            if (index >= args.length) {
                throw new IllegalArgumentException("missing value for " + flag);
            }
            return args[index];
        }
    }
}
