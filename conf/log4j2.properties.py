LOG4J2_PROPERTIES = """
# Set everything to WARN by default (Silences INFO logs)
rootLogger.level = WARN
rootLogger.appenderRef.stdout.ref = STDOUT

# Allow YOUR code to speak (Assumes you name your logger 'TraffyFloodETL')
logger.traffy.name = TraffyFloodETL
logger.traffy.level = INFO

# Standard Console Appender
appender.console.type = Console
appender.console.name = STDOUT
appender.console.target = SYSTEM_ERR
appender.console.layout.type = PatternLayout
# Format: Time Level [Thread] Logger: Message
appender.console.layout.pattern = %d{yyyy-MM-dd HH:mm:ss} %p %c{1}: %m%n
"""