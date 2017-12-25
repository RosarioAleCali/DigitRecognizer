#ifndef MODEL_FILE_H
#define MODEL_FLE_H

#include "daal.h"
#include <fstream>
#include <vector>

class ModelFileWriter 
{
    public:
        ModelFileWriter(const char* filename) 
            : fileBuffer(filename, std::ios::out|std::ios::binary) { }

        ~ModelFileWriter()
        {
            fileBuffer.close();
        }

        void serializeToFile(daal::services::SharedPtr<daal::algorithms::multi_class_classifier::Model> model)
        {
            // Serialize
            daal::data_management::InputDataArchive archive;
            model->serialize(archive);
            size_t length = archive.getSizeOfArchive();
            std::vector<daal::byte> data(length);
            archive.copyArchiveToArray(&data[0], length);

            // Write to file
            if (fileBuffer.is_open()) {
                fileBuffer.write((char*)&data[0], length);
            }
        }

    private:
        std::ofstream fileBuffer;
};

class ModelFileReader
{
    public:
        ModelFileReader(const char* filename)
            : fileBuffer(filename, std::ios::in|std::ios::binary|std::ios::ate) { }

        ~ModelFileReader()
        {
            fileBuffer.close();
        }

        void deserializeFromFile(daal::services::SharedPtr<daal::algorithms::multi_class_classifier::Model>& model)
        {
            if (fileBuffer.is_open()) {
                // Read raw bytes from file
                std::streampos end = fileBuffer.tellg();
                fileBuffer.seekg(0, std::ios::beg);
                size_t fileSize = static_cast<size_t>(end);
                std::vector<daal::byte> data(fileSize);
                if (fileBuffer.read((char*)&data[0], fileSize)) {
                    // Deserialize
                    daal::data_management::OutputDataArchive archive(&data[0], fileSize);
                    model->deserialize(archive);
                }
            }
        }

    private:
        std::ifstream fileBuffer;
};

#endif /*MODEL_FLE_H*/
