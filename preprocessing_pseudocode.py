#Define speaker names as they are formatted in the transcripts.
DEFINE interviewer_label AS REGEX MATCHING "Interviewer" (case insensitive)
DEFINE interviewee_label AS REGEX MATCHING "Interviewee" (case insensitive)

#Define folder paths. The input folder contains original transcripts. 
#The output folder will contain the processed transcripts.  
SET input_folder TO '/filepath/Transcript'
SET output_folder TO '/filepath/Cleaned_Transcript'

#Loop through the all files in the input folder.
FOR EACH filename IN input_folder:
    SET input_file TO input_folder + filename
    SET output_file TO output_folder + filename

    #Reads the contents of the file and stores each line separately. 
	OPEN input_file AS file
    READ lines FROM file USING readlines
    CLOSE file

    #Create a flag for tracking the interviewer. 
    SET interviewer TO False
    #Create a data structure to store the cleaned lines of text. 
    INITIALIZE modified_lines AS empty list

    #Loop through each line in the transcript.
    FOR EACH line IN transcript:
        #Remove extra whitespace. 
        STRIP line
        #Check if the line starts with the interviewer label.
        IF line MATCHES interviewer_label THEN:
            #If the line is language from the interviewer, set the flag to true. 
            SET interviewer TO True
            CONTINUE to next line # Do not store this line and move to the next one. 
        END IF

        #Check if line strats with language from the interviewee. 
        IF line MATCHES interviewee_label THEN:
            #If yes, set the flag to false and apply any cleaning steps to the line. 
            SET interviewer TO False
            REMOVE interviewee_label FROM line
        END IF

        #If a line does not start with a speaker label, check if it is interviewee language.
        IF interviewer IS False THEN:
            #If yes, save the text. 
            APPEND line TO modified_lines
        END IF
    END FOR

    #Combine the filtered lines of text back into a single document. 
    #Write the text to an output file. 
    JOIN modified_lines INTO content WITH spaces
    OPEN output_file AS file
    WRITE content TO file
    CLOSE file
END FOR
