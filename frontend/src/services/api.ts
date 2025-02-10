import axios from "axios";
import CodeRefactorRequest from "../interfaces/CodeRefactorRequest";
import CodeRefactorResponse from "../interfaces/CodeRefactorResponse";

const API_BASE_URL = "http://127.0.0.1:8000";

export const refactorCode = async (requestData: CodeRefactorRequest): Promise<CodeRefactorResponse> => {
    try {
        const response = await axios.post<CodeRefactorResponse>(`${API_BASE_URL}/refactor`, requestData);
        return response.data;
    } catch (error) {
        throw error;
    }
};