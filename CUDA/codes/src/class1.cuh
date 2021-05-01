class dataset{
public:
    int * dev_data;
    int * host_data;
    int size;
    void test_cpu(int n);
    void test_gpu(int n);
};