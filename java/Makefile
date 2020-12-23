DIR := src
SOURCE := source.list
OUT := target

javac: preinit
	javac -d $(OUT)  @$(DIR)/$(SOURCE)
	rm -f $(DIR)/$(SOURCE)

preinit:
	@find $(DIR) -name "*.java" > $(DIR)/$(SOURCE) 

clean:
	rm -rf $(OUT)