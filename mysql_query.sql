use kaivv_infer_results;

CREATE TABLE RESULT (
    ID INT NOT NULL AUTO_INCREMENT,
    FILE_DIR VARCHAR(100) NOT NULL,
    RESULT_STT TEXT,infers
    RESULT_AUDIO JSON,
    RESULT_IMAGE JSON,
    RESULT_SENTENCE JSON,
    PRIMARY KEY (ID)
);

CREATE TABLE `infers` (
  `id` int NOT NULL,
  `result_id` int NOT NULL,
  `timestep` int DEFAULT NULL,
  `audio_infer` int DEFAULT NULL,
  `sent_infer` int DEFAULT NULL,
  `sent_content` text,
  PRIMARY KEY (`id`),
  KEY `infers_FK` (`result_id`),
  CONSTRAINT `infers_FK` FOREIGN KEY (`result_id`) REFERENCES `result` (`ID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='infers collection!'

select * from result;

ALTER TABLE infers ADD CONSTRAINT infers_FK FOREIGN KEY(result_id) REFERENCES result(id); 













































































































































































